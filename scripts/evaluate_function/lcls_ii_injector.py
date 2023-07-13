import time
import numpy as np

from epics import caget, caput, caget_many
from scripts.utils.image_processing import get_beam_data
from time import sleep


def measure_beamsize(inputs):
    roi = inputs["roi"]
    screen = inputs["screen"]
    threshold = inputs["threshold"]

    if inputs["background"] is not None:
        background_image = np.load(inputs["background"])
    else:
        background_image = None

    # set PVs
    for k, v in inputs.items():
        print(f'CAPUT {k} {v}')
        caput(k, v)

    sleep(1.0)

    # get image data
    img, nx, ny = caget_many([
        f"{screen}:image1:ArrayData",
        f"{screen}:image1:ArraySize1_RBV",
        f"{screen}:image1:ArraySize0_RBV"
    ])

    # reshape image and subtract background image (set negative values to zero)
    if background_image is not None:
        img = img.reshape(nx, ny)
        img = img - background_image
        img = np.where(img >= 0, img, 0)

    results = get_beam_data(img, roi, threshold)
    current_time = time.time()
    results["time"] = current_time

    return results

