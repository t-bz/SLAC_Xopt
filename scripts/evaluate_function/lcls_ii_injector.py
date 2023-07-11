import time

from epics import caget, caput, caget_many
from scripts.utils.image_processing import get_beam_data
from time import sleep


def measure_beamsize(inputs):
    roi = inputs["roi"]
    screen = inputs["screen"]
    threshold = inputs["threshold"]
    camera = inputs["camera"]

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
    img = img.reshape(nx, ny)

    results = get_beam_data(img, roi, threshold)

    # get the camera resolution in meters/pixel
    resolution = caget(f"{camera}:RESOLUTION") * 1e-6

    # convert beam size results to meters
    results['Sx'] = results['Sx'] * resolution
    results['Sy'] = results['Sy'] * resolution

    current_time = time.time()
    results["time"] = current_time

    return results

