from typing import List, Tuple

import torch.utils.data as tdata
from toolbox.dataset import GenericDataset, GenericLMDBDataset
from toolbox.dataset.hazy_dataset import DHazeDataset
from toolbox.imageOperators import GaussianBlurFFT as BlurConvolution
from toolbox.imageOperators import GaussianNoise, Kernels

from degradation import (
    BlurApplication,
    ColorToGrayDegradation,
    DegradationFunction,
    tanh_saturation,
    tanh_saturation_hard,
    tanh_saturation_normalised,
    tanh_saturation_normalised_hard,
)

DATA_ROOT = "/data/projects_backup/training_denoisers_younes/train_denoisers/BSR/BSDS500/data/images/all/"
BSD_TRAIN_PATH = DATA_ROOT + "train/0/"
IMNET_TRAIN_PATH = "/home/youyoun/Projects/MonotoneInverse/Imagenet/train.lmdb"
TEST_DATA = DATA_ROOT + "test/0/"

HAZY_DATASET_TRAIN = "/home/youyoun/Projects/MonotoneInverse/datasets/NYU"
HAZY_DATASET_TEST = "/home/youyoun/Projects/MonotoneInverse/datasets/Middlebury"

NUM_WORKERS = 6

random_blur_parameters = [
    (2.0633368492126465, 1.7180869579315186),
    (2.4405579566955566, 2.4832868576049805),
    (2.0951995849609375, 2.328171730041504),
    (0.8302425146102905, 1.8501242399215698),
    (1.3940463066101074, 1.0478699207305908),
    (1.9406932592391968, 0.6032775640487671),
    (2.4030351638793945, 1.3876314163208008),
    (1.7948418855667114, 1.0497373342514038),
    (1.0755391120910645, 2.110309600830078),
    (1.9366323947906494, 1.9609949588775635),
    (2.307396173477173, 0.6718982458114624),
    (1.5582815408706665, 0.9032354354858398),
    (1.3387272357940674, 0.6171562671661377),
    (0.5293096303939819, 1.692731499671936),
    (1.9672998189926147, 0.9757387638092041),
    (2.2329468727111816, 2.18892502784729),
    (0.8333008289337158, 0.988949179649353),
    (0.6217002868652344, 0.9062840938568115),
    (0.7757526636123657, 1.1312791109085083),
    (1.8043538331985474, 2.4241650104522705),
    (2.1500930786132812, 1.7860645055770874),
    (2.4712393283843994, 2.162670612335205),
    (2.0018129348754883, 2.388181686401367),
    (0.5571883916854858, 0.653221607208252),
    (2.1269235610961914, 1.1473824977874756),
    (1.9672659635543823, 2.3575844764709473),
    (1.5214476585388184, 1.0076978206634521),
    (1.8875067234039307, 1.5889214277267456),
    (0.5062227249145508, 1.5569651126861572),
    (0.9009115695953369, 1.219071626663208),
    (1.9211270809173584, 2.224088668823242),
    (0.659963846206665, 2.410557508468628),
    (2.013568878173828, 2.0648436546325684),
    (0.6788665056228638, 2.1617136001586914),
    (1.0433158874511719, 1.3308769464492798),
    (1.6791713237762451, 1.5513486862182617),
    (0.9586600065231323, 1.2139496803283691),
    (0.7357637882232666, 2.2789793014526367),
    (2.1179890632629395, 1.6685049533843994),
    (0.7746084928512573, 1.0810520648956299),
    (2.050229072570801, 0.7656675577163696),
    (0.7951182126998901, 1.7912479639053345),
    (1.455429196357727, 1.2957485914230347),
    (0.6482206583023071, 0.561577320098877),
    (1.6638091802597046, 1.8376175165176392),
    (1.15380859375, 0.8746960163116455),
    (0.793063759803772, 1.0566136837005615),
    (1.9559847116470337, 0.9095062017440796),
    (1.8106824159622192, 1.675064206123352),
    (0.8607937097549438, 2.4029150009155273),
    (0.7898215055465698, 2.016326427459717),
    (2.4410033226013184, 1.967301607131958),
    (0.514784574508667, 1.6513512134552002),
    (1.3508177995681763, 2.0775961875915527),
    (2.0884313583374023, 1.9525824785232544),
    (1.8193005323410034, 1.5793207883834839),
    (1.0536904335021973, 2.4426093101501465),
    (2.2742881774902344, 1.2075384855270386),
    (1.0550695657730103, 1.5897585153579712),
    (1.7512060403823853, 0.9418027400970459),
    (1.8871915340423584, 0.8420878648757935),
    (0.7263935804367065, 2.3000831604003906),
    (0.9831657409667969, 1.7664707899093628),
    (0.7747118473052979, 2.490980625152588),
    (1.1926565170288086, 2.1679792404174805),
    (0.6164947748184204, 1.8688626289367676),
    (1.333204984664917, 1.7951520681381226),
    (1.7272777557373047, 1.8546326160430908),
    (2.3112740516662598, 0.9610354900360107),
    (0.895519495010376, 1.087597370147705),
    (2.4170424938201904, 1.8988840579986572),
    (1.21469247341156, 0.6581426858901978),
    (0.848418116569519, 2.451570987701416),
    (2.314509391784668, 1.1441092491149902),
    (1.331782579421997, 1.7591440677642822),
    (2.034010887145996, 2.246674060821533),
    (1.9156453609466553, 2.235811948776245),
    (0.5203491449356079, 2.3471622467041016),
    (2.389768123626709, 1.4430091381072998),
    (1.0062326192855835, 2.4257125854492188),
    (1.7988073825836182, 1.5721337795257568),
    (2.2104482650756836, 2.2536890506744385),
    (1.2550922632217407, 1.3222191333770752),
    (2.13089656829834, 2.4464948177337646),
    (1.2548943758010864, 1.2847599983215332),
    (1.3867242336273193, 2.4545412063598633),
    (2.4068493843078613, 1.6356772184371948),
    (2.4532809257507324, 0.5639891624450684),
    (2.2001843452453613, 1.3820738792419434),
    (1.214113473892212, 0.82133948802948),
    (2.0254626274108887, 2.482830762863159),
    (1.159163475036621, 1.8844109773635864),
    (2.043058156967163, 1.5404571294784546),
    (1.1796653270721436, 1.112813115119934),
    (0.6373535394668579, 1.973489761352539),
    (1.2934317588806152, 1.639532208442688),
    (1.932421326637268, 0.8072466850280762),
    (1.3026938438415527, 1.712146520614624),
    (2.3906607627868652, 1.2243129014968872),
    (2.3873348236083984, 2.4467272758483887),
]
random_thetas = [
    2.6105,
    -2.6713,
    0.6381,
    -1.3073,
    -0.7573,
    -2.4788,
    0.0099,
    -2.9784,
    2.9549,
    0.3904,
]
random_alphas = [
    1.1413,
    0.8773,
    1.2033,
    1.3310,
    1.1060,
    1.1970,
    1.3951,
    0.9562,
    1.0546,
    0.8387,
]


def get_alpha_gabor(
    blur_parameters: List[BlurConvolution], thetas: List[float], alphas: List[float]
) -> List[BlurConvolution]:
    blurs = [
        BlurConvolution(9, Kernels.GABOR, (b_1, b_2), frequency=1.0, theta=theta)
        for ((b_1, b_2), theta) in zip(blur_parameters, thetas)
    ]
    for i, b in enumerate(blurs):
        b.kernel = b.kernel * alphas[i]
    return blurs


GAUSSIAN_KERNEL_SIZE = 17

degrad_blur_parameters = {
    "None": [],
    "single": [BlurConvolution(3, Kernels.UNIFORM, 0.0)],
    "uni7x7": [BlurConvolution(7, Kernels.UNIFORM, 0.0)],
    "double": [
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (1.0, 0.5)),
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (0.5, 1.0)),
    ],
    "two": [
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (2.0, 1.0)),
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (1.0, 2.0)),
    ],
    "three": [
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (2.0, 1.0)),
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (1.0, 2.0)),
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (1.0, 1.0)),
    ],
    "hundred": [
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (b_1, b_2))
        for (b_1, b_2) in random_blur_parameters
    ],
    "ten": [
        BlurConvolution(GAUSSIAN_KERNEL_SIZE, Kernels.GAUSSIAN, (b_1, b_2))
        for (b_1, b_2) in random_blur_parameters[:10]
    ],
    "tengabor": [
        BlurConvolution(
            GAUSSIAN_KERNEL_SIZE, Kernels.GABOR, (b_1, b_2), frequency=1.0, theta=theta
        )
        for ((b_1, b_2), theta) in zip(random_blur_parameters[:10], random_thetas[:10])
    ],
    "tengaboralpha": get_alpha_gabor(
        random_blur_parameters[:10], random_thetas[:10], random_alphas[:10]
    ),
    "mvt1": [BlurConvolution(30, Kernels.TYPE_D, 0.0)],
    "mvt5": [
        BlurConvolution(30, Kernels.TYPE_D, 0.0),
        BlurConvolution(30, Kernels.TYPE_E, 0.0),
        BlurConvolution(30, Kernels.TYPE_F, 0.0),
        BlurConvolution(30, Kernels.TYPE_A, 0.0),
        BlurConvolution(30, Kernels.TYPE_G, 0.0),
    ],
    "mvt1_small": [BlurConvolution(9, Kernels.MVT_1, 0.0)],
    "mvt5_small": [
        BlurConvolution(9, Kernels.MVT_1, 0.0),
        BlurConvolution(9, Kernels.MVT_2, 0.0),
        BlurConvolution(9, Kernels.MVT_3, 0.0),
        BlurConvolution(9, Kernels.MVT_4, 0.0),
        BlurConvolution(9, Kernels.MVT_5, 0.0),
    ],
}


def get_dataset(
    dataset_name: str,
    test_dataset_name: str,
    crop_size: int,
    batch_size: int,
    blur_parameters: List[BlurConvolution],
    n_images: int = 0,
    blur_app_type: BlurApplication = BlurApplication.SUM,
    random_crop: bool = False,
    normalize: bool = False,
    return_test_only: bool = False,
    colorized: bool = False,
    no_crop: bool = False,
    use_hard_saturation: bool = False,
) -> Tuple[tdata.Dataset, tdata.DataLoader, tdata.Dataset, tdata.DataLoader]:
    if not no_crop:
        train_dataset_transforms = [
            ("randomcrop", {"size": crop_size, "pad_if_needed": True})
            if random_crop
            else ("centercrop", {"size": crop_size})
        ]
        test_dataset_transforms = [("centercrop", {"size": crop_size})]
    else:
        train_dataset_transforms = []
        test_dataset_transforms = []
    if normalize:
        train_dataset_transforms.append(("normalize", {"mean": 0.5, "std": 0.5}))
        test_dataset_transforms.append(("normalize", {"mean": 0.5, "std": 0.5}))

    if blur_app_type == BlurApplication.GRAY:
        degrad_fn = ColorToGrayDegradation()
    else:
        if normalize:
            tanh_saturation_fn = (
                tanh_saturation_normalised_hard
                if use_hard_saturation
                else tanh_saturation_normalised
            )
        else:
            tanh_saturation_fn = (
                tanh_saturation_hard if use_hard_saturation else tanh_saturation
            )
        degrad_fn = DegradationFunction(
            Li=blur_parameters,
            S=tanh_saturation_fn,
            alphai=0.0 if blur_app_type == BlurApplication.SL else 1.0,
            type_=blur_app_type,
        )

    if blur_app_type == BlurApplication.COMPOSE:
        print("Using Composition L^T S (Lx) blur")
    elif blur_app_type == BlurApplication.SUM:
        print("Using Sum L^T S(Lx) blur")
    elif blur_app_type == BlurApplication.SL:
        print("Using Sum S(Lx) blur")
    elif blur_app_type == BlurApplication.GRAY:
        print("Using Color to Gray blur")
    else:
        raise ValueError("Unknown blur application type")

    if not return_test_only:
        if dataset_name == "bsd":
            ds = GenericDataset(
                BSD_TRAIN_PATH,
                n_images,
                degrad_fn,
                augments=train_dataset_transforms,
                load_in_memory=True,
                colorized=colorized,
            )
            dl = tdata.DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == "imagenet":
            ds = GenericLMDBDataset(
                IMNET_TRAIN_PATH,
                degrad_fn,
                n_images,
                augments=train_dataset_transforms,
                load_in_memory=False,
            )
            dl = tdata.DataLoader(
                ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
            )
        elif dataset_name == "hazy":
            ds = DHazeDataset(HAZY_DATASET_TRAIN, n_images, train_dataset_transforms)
            dl = tdata.DataLoader(ds, batch_size=batch_size, shuffle=True)
        else:
            raise ValueError("Unknown dataset name")
    else:
        ds = None
        dl = None
    # print("Loading only test set on memory")
    if test_dataset_name == "bsd" or dataset_name == "imagenet":
        test_ds = GenericDataset(
            TEST_DATA,
            n_images,
            degrad_fn,
            augments=test_dataset_transforms,
            load_in_memory=True,
            colorized=colorized,
        )
        test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    elif test_dataset_name == "hazy":
        test_ds = DHazeDataset(HAZY_DATASET_TEST, n_images, test_dataset_transforms)
        test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unknown dataset name")
    return ds, dl, test_ds, test_dl


def get_noisy_dataset(
    dataset_name: str,
    test_dataset_name: str,
    gaussian_mean: float,
    gaussian_std: float,
    crop_size: int,
    batch_size: int,
    n_images: int = 0,
    random_crop: bool = False,
    normalize: bool = False,
    return_test_only: bool = False,
    colorized: bool = False,
    no_crop: bool = False,
) -> Tuple[tdata.Dataset, tdata.DataLoader, tdata.Dataset, tdata.DataLoader]:
    if not no_crop:
        train_dataset_transforms = [
            ("randomcrop", {"size": crop_size, "pad_if_needed": True})
            if random_crop
            else ("centercrop", {"size": crop_size})
        ]
        test_dataset_transforms = [("centercrop", {"size": crop_size})]
    else:
        train_dataset_transforms = []
        test_dataset_transforms = []
    if normalize:
        train_dataset_transforms.append(("normalize", {"mean": 0.5, "std": 0.5}))
        test_dataset_transforms.append(("normalize", {"mean": 0.5, "std": 0.5}))

    degrad_fn = GaussianNoise(gaussian_mean, gaussian_std)

    if not return_test_only:
        if dataset_name == "bsd":
            ds = GenericDataset(
                BSD_TRAIN_PATH,
                n_images,
                degrad_fn,
                augments=train_dataset_transforms,
                load_in_memory=False,
                colorized=colorized,
            )
            dl = tdata.DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == "imagenet":
            ds = GenericLMDBDataset(
                IMNET_TRAIN_PATH,
                degrad_fn,
                n_images,
                augments=train_dataset_transforms,
                load_in_memory=False,
            )
            dl = tdata.DataLoader(
                ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
            )
        elif dataset_name == "hazy":
            ds = DHazeDataset(HAZY_DATASET_TRAIN, n_images, train_dataset_transforms)
            dl = tdata.DataLoader(ds, batch_size=batch_size, shuffle=True)
        else:
            raise ValueError("Unknown dataset name")

    else:
        ds = None
        dl = None
    # print("Loading only test set on memory")
    if test_dataset_name == "bsd" or dataset_name == "imagenet":
        test_ds = GenericDataset(
            TEST_DATA,
            n_images,
            degrad_fn,
            augments=test_dataset_transforms,
            load_in_memory=True,
            colorized=colorized,
        )
        test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    elif test_dataset_name == "hazy":
        test_ds = DHazeDataset(HAZY_DATASET_TEST, n_images, test_dataset_transforms)
        test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unknown dataset name")
    return ds, dl, test_ds, test_dl
