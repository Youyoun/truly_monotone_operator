import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as ttransforms
from piq import PieAPP, StyleLoss
from toolbox.jacobian import MonotonyRegularizationShift, PenalizationMethods

import GANModules as networks


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


class Training:
    def __init__(self, generator, discriminator, opt, linear_model=None):
        self.gen = generator
        self.optimG = optimizers.Adam(generator.parameters(), lr=opt.learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimG,
            "min",
            patience=5,
            threshold=1e-4,
            verbose=True,
        )
        if not (opt.deactivate_gan or opt.use_monotony):
            self.disc = discriminator
            self.optimD = optimizers.Adam(
                discriminator.parameters(), lr=opt.learning_rate
            )
            self.criterionGAN = networks.GANLoss("vanilla", 0.99, 0.01).to(opt.device)
        if opt.loss == "l1":
            self.criterionL1 = nn.L1Loss()
        elif opt.loss == "l2":
            self.criterionL1 = nn.MSELoss()
        elif opt.loss == "pieapp":

            def pieapp_normalized(input_, target):
                pieapp = PieAPP(enable_grad=True)
                input_ = (input_ + 1) / 2
                target = (target + 1) / 2
                return torch.sqrt(pieapp(input_, target) ** 2 + 1) - 1

            self.criterionL1 = pieapp_normalized
        elif opt.loss == "style":
            print("Using style loss")
            self.style_loss = StyleLoss()
            self.criterionL1 = nn.L1Loss()
        else:
            raise NotImplementedError("Only l1 and l2 losses are implemented")
        if opt.use_monotony:
            # self.monotonyG = networks.MonotonyLoss(generator, opt)
            self.monotonyG = MonotonyRegularizationShift(
                PenalizationMethods.OPTPOWERNOALPHA,
                opt.eps_monotony,
                None,
                opt.max_iter_power_method,
            )
        self.opt = opt

        self.linear_model = linear_model

    def monotony_forward(self, x):
        if self.linear_model is not None:
            return self.linear_model.forward_t(self.gen(x))
        else:
            return self.gen(x)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backwardG(self, input_, target, prediction):
        # 3 losses, GAN, L1, Monotony
        losses = {}

        # Second, G(A) = B
        loss_ = self.criterionL1(prediction, target) * self.opt.lambda_l1
        losses["L1"] = loss_.item()

        if self.opt.loss == "style":
            style_loss = self.style_loss(prediction, target)
            losses["Style"] = style_loss.item() * self.opt.lambda_style
            loss_ = loss_ + style_loss * self.opt.lambda_style

            if self.opt.lambda_tv > 0:
                tv_loss = calc_TV_Loss(prediction)
                losses["TV"] = tv_loss.item() * self.opt.lambda_tv
                loss_ = loss_ + tv_loss * self.opt.lambda_tv

        if self.opt.use_monotony:
            eta = torch.rand(1).to(self.opt.device)
            if self.opt.monotony_crop:
                crop_fn = ttransforms.CenterCrop(self.opt.monotony_crop)
                loss_G_Mon, _ = self.monotonyG(
                    self.monotony_forward,
                    (eta * crop_fn(input_) + (1 - eta) * crop_fn(target))[:1],
                )
            else:
                loss_G_Mon, _ = self.monotonyG(
                    self.monotony_forward, (eta * input_ + (1 - eta) * target)[:1]
                )
            losses["Mon"] = loss_G_Mon.item()
            losses["Lambda_mon"] = self.opt.lambda_monotony
            loss_ = loss_ + loss_G_Mon * self.opt.lambda_monotony

        if not self.opt.deactivate_gan and not self.opt.use_monotony:
            fake_AB = torch.cat((input_, prediction), 1)
            pred_fake = self.disc(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            losses["GAN"] = loss_G_GAN.item()
            loss_ = loss_ + loss_G_GAN

        loss_.backward()

        return losses

    def backwardD(self, input_, target, prediction):
        if not self.opt.deactivate_gan and not self.opt.use_monotony:
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat(
                (input_, prediction), 1
            )  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.disc(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((input_, target), 1)
            pred_real = self.disc(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            return loss_D.item()
        else:
            return 0

    def optimall(self, input_, target, prediction):
        # update D
        losses = {}
        if not self.opt.use_monotony and not self.opt.deactivate_gan:
            Training.set_requires_grad(self.disc, True)  # enable backprop for D
            self.optimD.zero_grad()  # set D's gradients to zero
            losses["D"] = self.backwardD(
                input_, target, prediction
            )  # calculate gradients for D
            self.optimD.step()  # update D's weights
            Training.set_requires_grad(
                self.disc, False
            )  # D requires no gradients when optimizing G
        # update G
        self.optimG.zero_grad()  # set G's gradients to zero
        g_losses = self.backwardG(
            input_, target, prediction
        )  # calculate graidents for G
        self.optimG.step()  # udpate G's weights
        losses.update(g_losses)
        return losses
