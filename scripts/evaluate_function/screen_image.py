import time

import numpy as np
import pandas as pd
import h5py

from scripts.utils.image_processing import get_beam_data
from time import sleep

TESTING = True

if not TESTING:
    from epics import caput, caget_many


def get_raw_image(screen_name):
    # get image data
    if TESTING:
        img = np.load(screen_name)
        resolution = 1.0
    else:
        img, nx, ny, resolution = caget_many([
            f"{screen_name}:IMAGE",
            f"{screen_name}:ROI_XNP",
            f"{screen_name}:ROI_YNP",
            f"{screen_name}:RESOLUTION"
        ])
        img = img.reshape(ny, nx)

    return img, resolution


def measure_background(screen_name, n_measurements: int = 20, filename: str = None,):
    filename = filename or f"{screen_name}_background".replace(":", "_")

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
    screen = inputs.pop("screen")

    # NOTE: the defaults specified here are not tracked by Xopt!
    roi = inputs.pop("roi", None)
    n_shots = inputs.pop("n_shots", 1)
    bb_half_width = inputs.pop("bb_half_width", 2.0)
    visualize = inputs.pop("visualize", False)
    save_img_location = inputs.pop("save_img_location", None)
    sleep_time = inputs.pop("sleep_time", 1.0)
    min_log_intensity = inputs.pop("min_log_intensity", 0.0)
    extra_pvs = inputs.pop("extra_pvs", [])

    background = inputs.pop("background", None)
    if background is not None:
        background_image = np.load(background)
    else:
        background_image = None

    # set PVs
    if not TESTING:
        for k, v in inputs.items():
            print(f'CAPUT {k} {v}')
            caput(k, v)

    sleep(sleep_time)

    data = []
    images = []
    results = {}
    start_time = time.time()
    for _ in range(n_shots):
        img, resolution = get_raw_image(screen)

        # reshape image and subtract background image (set negative values to zero)
        if background_image is not None:
            img = img - background_image
            img = np.where(img >= 0, img, 0)

        s = time.time()
        results = get_beam_data(
            img, roi, min_log_intensity=min_log_intensity,
            bb_half_width=bb_half_width, visualize=visualize,
            n_restarts=10
        )
        # print(f"fitting time:{time.time() - s}")

        # add measurements of extra pvs to results
        if not TESTING:
            extra_results = dict(zip(extra_pvs, caget_many(extra_pvs)))
            results = results | extra_results

        # convert beam size results to meters
        if results["Sx"] is not None:
            results['Sx'] = results['Sx'] * resolution
            results['Sy'] = results['Sy'] * resolution

        current_time = time.time()
        results["time"] = current_time

        sleep(1.0)

        data += [results]
        images += [img]

    if n_shots == 1:
        outputs = data[0]
    else:
        # collect results into lists
        outputs = pd.DataFrame(data).reset_index().to_dict(orient='list')
        outputs.pop("index")

        # create numpy arrays from lists
        outputs = {key: list(np.array(ele)) for key, ele in outputs.items()}

    # if specified, save image data to location based on time stamp
    save_filename = f"{save_img_location}/{start_time}.h5"
    with h5py.File(save_filename, "w") as hf:
        dset = hf.create_dataset("images", data=np.array(images))
        for name, val in (outputs | inputs).items():
            dset.attrs[name] = val

    outputs["save_filename"] = save_filename

    #print(outputs)
    return outputs

