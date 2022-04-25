from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np


def show_images(images: List, names: List = None, size: Tuple[int] = (5, 5), cmap="gray"):
    f, axarr = plt.subplots(1, len(images), figsize=size)
    # case one image provided
    if len(images) == 1:
        pixels = np.array(images[0])
        axarr.imshow(pixels, cmap=cmap)

        axarr.set_yticks(range(4))
        axarr.set_yticklabels(["City 1", "City 2", "City 3", "City 4"])
        axarr.set_xticks(range(5))
        axarr.set_xticklabels(["Wind speed", "Direction", "Temperature", "Dew point", "Air pressure"], rotation=45)

        if names:
            axarr.set_title(names[0])

        #axarr.axis("off")
        return

    # case multiple images provided
    for i, image in enumerate(images):
        pixels = np.array(image)
        axarr[i].imshow(pixels, cmap=cmap)

        axarr[i].set_yticks(range(4))
        axarr[i].set_yticklabels(["City 1", "City 2", "City 3", "City 4"])
        axarr[i].set_xticks(range(5))
        axarr[i].set_xticklabels(["Wind speed", "Direction", "Temperature", "Dew point", "Air pressure"], rotation=45)

        if names:
            axarr[i].set_title(names[i])

    f.tight_layout()
        #axarr[i].axis("off")