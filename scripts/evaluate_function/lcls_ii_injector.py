import time
import numpy as np

from epics import caget, caput, caget_many
from scripts.utils.image_processing import get_beam_data
from time import sleep


def get_raw_image(screen_name):
    # get image data
    img, nx, ny = caget_many([
        f"{screen_name}:Image:ArrayData",
        f"{screen_name}:Image:ArraySize1_RBV",
        f"{screen_name}:Image:ArraySize0_RBV"
    ])
    img = img.reshape(nx, ny)

    return img


def measure_background(screen_name, n_measurements:int = 20, filename:str = None,):
    filename = filename or f"{screen_name}_background"
    filename += ".npy"

    images = []
    for i in range(n_measurements):
        images += [get_raw_image(screen_name)]
        sleep(0.1)

    # return average
    images = np.stack(images)
    mean = images.mean(axis=0)

    # save average
    np.save(filename, mean)

    return mean


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

    img = get_raw_image(screen)

    # reshape image and subtract background image (set negative values to zero)
    if background_image is not None:
        img = img - background_image
        img = np.where(img >= 0, img, 0)

    results = get_beam_data(img, roi, threshold)
    current_time = time.time()
    results["time"] = current_time

    return results

