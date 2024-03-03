import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.animation


def save_json(js, filepath, indent=None):
    with open(filepath, 'w') as f:
        json.dump(js, f, indent=indent)


def get_path(id, dir, suffix=""):
    Path(dir).mkdir(parents=True, exist_ok=True)
    filename = str(id)
    if suffix: filename += f"_{suffix}"
    return os.path.join(dir, filename)


def save_plot_args(args, dir, suffix=""):
    path = get_path(args.id, dir, suffix)
    plt.savefig(path, bbox_inches='tight')
    save_json(vars(args), f"{path}.json", indent=4)


def animate(imgs):
    
    n_fm = len(imgs)
    n_im = len(imgs[0])
    ncols = min(n_im, 5)
    nrows = int(np.ceil(n_im / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    frames = []
    for k in range(n_im):
        im = imgs[0][k]
        ax = axs[k // ncols, k % ncols] if nrows > 1 else axs[k]
        fm = ax.imshow(im)
        frames.append(fm)
        ax.axis("off")
    fig.tight_layout()
    plt.close()

    def update(i):
        for k in range(n_im):
            frames[k].set_array(imgs[i][k])
        return frames

    return matplotlib.animation.FuncAnimation(fig, update, frames=n_fm, interval=50, repeat=False)