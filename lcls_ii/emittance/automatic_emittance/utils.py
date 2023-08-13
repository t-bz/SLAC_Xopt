from xopt import VOCS
import numpy as np

from scripts.evaluate_function.screen_image import measure_beamsize


## import variable ranges
import pandas as pd
filename = "../../variables.csv"
VARIABLE_RANGES = pd.read_csv(filename, index_col=0, header=None).T.to_dict(orient='list')
SCREEN_NAME = "OTRS:HTR:330"

TUNING_VARIABLES = ["QUAD:HTR:120:BCTRL","QUAD:HTR:140:BCTRL","QUAD:HTR:300:BCTRL"]
SCAN_VARIABLE = "QUAD:HTR:320:BCTRL"
QUAD_LENGTH = 1.0 # m
DRIFT_LENGTH = 1.0 # m
BEAM_ENERGY = 0.135 # GeV
PV_TO_INTEGRATED_GRADIENT = 1.0 # kG
ROI = None

MEASUREMENT_OPTIONS = {
    "screen": SCREEN_NAME,
    "background": None,
    "roi": ROI,
    "bb_half_width": 3.0, # half width of the bounding box in terms of std
    "visualize": True
}

IMAGE_CONSTRAINTS = {
    "bb_penalty": ["LESS_THAN", 0.0],
    "log10_total_intensity": ["GREATER_THAN", 4]
}

# define function to measure the total size on OTR4
def eval_beamsize(input_dict):
    results = measure_beamsize(input_dict)
    results["S_x_mm"] = results["Sx"] * 1e3
    results["S_y_mm"] = results["Sy"] * 1e3

    #add total beam size
    results["total_size"] = np.sqrt(results["Sx"]**2 + results["Sy"]**2)
    return results
