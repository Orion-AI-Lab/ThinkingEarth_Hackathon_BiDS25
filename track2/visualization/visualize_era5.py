import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

def vis_data(
    pred, 
    truth,
    save_path,
    lat=None,
    lon=None,
    pred_title="Prediction",
    truth_title="Ground truth",
    cmap="twilight_shifted",
    projection="mollweide",
    diverging=False,
    figsize=(6, 4),
    vmax=None
    ):
    """
    Visualization tool to plot a comparison between ground truth and prediction
    pred: 2d array
    truth: 2d array
    cmap: colormap
    projection: "mollweide", "hammer", "aitoff" or None
    """
    
    assert len(pred.shape) == 2
    assert len(truth.shape) == 2
    assert pred.shape == truth.shape

    H, W = pred.shape
    if (lat is None) or (lon is None):
        lon = np.linspace(-np.pi, np.pi, W)
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)
    Lon, Lat = np.meshgrid(lon, lat)

    # only normalize with the truth
    vmax = vmax or np.abs(truth).max()
    # vmax = vmax or max(np.abs(pred).max(), np.abs(truth).max())
    if diverging:
        vmin = -vmax
    else:
        vmin = 0.0

    # Create first figure for prediction
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(1, 1, 1, projection=projection)
    ax1.pcolormesh(Lon, Lat, pred, cmap=cmap) #, vmin=vmin, vmax=vmax)
    ax1.set_title(pred_title)
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.tight_layout()
    
    # Save prediction figure
    save_path_pred = f"{save_path}/pangu_pred.png"
    plt.savefig(save_path_pred)
    plt.close(fig1)

    # Create second figure for truth/target
    fig2 = plt.figure(figsize=figsize)
    ax2 = fig2.add_subplot(1, 1, 1, projection=projection)
    ax2.pcolormesh(Lon, Lat, truth, cmap=cmap)#, vmin=vmin, vmax=vmax)
    ax2.set_title(truth_title)
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.tight_layout()
    
    # Save truth/target figure
    save_path_target = f"{save_path}/pangu_target.png"
    plt.savefig(save_path_target)
    plt.close(fig2)