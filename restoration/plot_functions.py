import torch
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from restoration.utils import denorm


def plot_im(im, title=None, ax=None, clip=True):
    if ax is None:
        fig, ax = plt.subplots()
    if torch.is_tensor(im):
        if len(im.shape) > 2:
            ax.imshow(im.permute(1, 2, 0))
        else:
            ax.imshow(
                im,
                cmap="gray",
                vmin=-1 if clip else None,
                vmax=1 if clip else None,
            )
        ax.axis("off")
    else:
        ax.plot(im["||x_{k+1} - x_k||_2 / ||y||_2"])
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_results(results, clip=True):
    fig, ax = plt.subplots(4, 5, figsize=(4 * 4, 5 * 4), dpi=180)
    ax = ax.ravel()

    plot_im(*results["x"], ax[1], clip=clip)
    plot_im(*results["y"], ax[2], clip=clip)

    plot_im(*results["metrics_notmon"], ax[5], clip=clip)
    plot_im(*results["x_A_notmon_star"], ax[6], clip=clip)
    plot_im(*results["y_A_notmon"], ax[7], clip=clip)
    plot_im(*results["x_A_notmon"], ax[8], clip=clip)
    plot_im(*results["metrics_notmon_star"], ax[9], clip=clip)

    plot_im(*results["metrics_mon"], ax[10], clip=clip)
    plot_im(*results["x_A_mon_star"], ax[11], clip=clip)
    plot_im(*results["y_A_mon"], ax[12], clip=clip)
    plot_im(*results["x_A_mon"], ax[13], clip=clip)
    plot_im(*results["metrics_mon_star"], ax[14], clip=clip)

    plot_im(*results["metrics_lin"], ax[15], clip=clip)
    plot_im(*results["x_A_lin_star"], ax[16], clip=clip)
    plot_im(*results["y_A_lin"], ax[17], clip=clip)
    plot_im(*results["x_A_lin"], ax[18], clip=clip)
    plot_im(*results["metrics_lin_star"], ax[19], clip=clip)

    ax[0].remove()
    ax[3].remove()
    ax[4].remove()
    return fig, ax


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)
