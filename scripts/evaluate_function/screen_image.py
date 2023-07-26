import time
import numpy as np
import pandas as pd

from scripts.utils.image_processing import get_beam_data
from time import sleep

TESTING = True

if not TESTING:
    from epics import caput, caget_many


def get_raw_image(screen_name):
    # get image data
    if TESTING:
        img = np.load("../../../tests/test_img.npy")
        resolution = 1.0
    else:
        img, nx, ny, resolution = caget_many([
            f"{screen_name}:Image:ArrayData",
            f"{screen_name}:Image:ArraySize1_RBV",
            f"{screen_name}:Image:ArraySize0_RBV",
            f"{screen_name}:RESOLUTION"
        ])
        img = img.reshape(nx, ny)

    return img, resolution


def measure_background(screen_name, n_measurements: int = 20, filename: str = None,):
    filename = filename or f"{screen_name}_background".replace(":","_")

    images = []
    for i in range(n_measurements):
        images += [get_raw_image(screen_name)[0]]
        sleep(0.1)

    # return average
    images = np.stack(images)
    mean = images.mean(axis=0)

    if TESTING:
        mean = np.zeros_like(mean)

    # save average
    np.save(filename, mean)

    return mean


def measure_beamsize(inputs):
    roi = inputs.pop("roi")
    screen = inputs.pop("screen")
    threshold = inputs.pop("threshold")
    n_shots = inputs.pop("n_shots", 1)
    bb_half_width = inputs.pop("bb_half_width", 2.0)
    visualize = inputs.pop("visualize", False)

    if inputs["background"] is not None:
        background_image = np.load(inputs.pop("background"))
    else:
        background_image = None

    # set PVs
    if not TESTING:
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

        results = get_beam_data(
            img, roi, threshold, bb_half_width=bb_half_width, visualize=visualize
        )

        # convert beam size results to meters
        if results["Sx"] is not None:
            results['Sx'] = results['Sx'] * resolution
            results['Sy'] = results['Sy'] * resolution

        current_time = time.time()
        results["time"] = current_time

        data += [results]

    if n_shots == 1:
        outputs = data[0]
    else:
        # collect results into lists
        outputs = pd.DataFrame(data).reset_index().to_dict(orient='list')

        # create numpy arrays from lists
        outputs = {key: np.array(ele) for key, ele in outputs.items()}

    return outputs

