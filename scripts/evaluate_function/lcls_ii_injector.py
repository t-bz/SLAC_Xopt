import time
import numpy as np
import pandas as pd

from epics import caget, caput, caget_many
from scripts.utils.image_processing import get_beam_data
from time import sleep


def get_raw_image(screen_name):
    # get image data
    img, nx, ny, resolution = caget_many([
        f"{screen_name}:Image:ArrayData",
        f"{screen_name}:Image:ArraySize1_RBV",
        f"{screen_name}:Image:ArraySize0_RBV",
        f"{screen_name}:RESOLUTION"
    ])
    img = img.reshape(nx, ny)

    return img, resolution


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
    roi = inputs.pop("roi")
    screen = inputs.pop("screen")
    threshold = inputs.pop("threshold")
    n_shots = inputs.pop("n_shots", 1)

    if inputs["background"] is not None:
        background_image = np.load(inputs.pop("background"))
    else:
        background_image = None

    # set PVs
    for k, v in inputs.items():
        print(f'CAPUT {k} {v}')
        caput(k, v)

    sleep(1.0)

    data = []
    for _ in range(n_shots):
        img, resolution = get_raw_image(screen)

        # reshape image and subtract background image (set negative values to zero)
        if background_image is not None:
            img = img - background_image
            img = np.where(img >= 0, img, 0)

        results = get_beam_data(img, roi, threshold)

        # convert beam size results to meters
        results['Sx'] = results['Sx'] * resolution
        results['Sy'] = results['Sy'] * resolution

        current_time = time.time()
        results["time"] = current_time

        data += [results]

    outputs = pd.DataFrame(data)

    return outputs.to_dict()

