from xopt import VOCS
import numpy as np

from scripts.evaluate_function.screen_image import measure_beamsize


## import variable ranges
import pandas as pd
filename = "../../variables.csv"
VARIABLE_RANGES = pd.read_csv(filename, index_col=0, header=None).T.to_dict(orient='list')
SCREEN_NAME = "OTRS:HTR:330"

TUNING_VARIABLES = ["QUAD:HTR:120:BCTRL","QUAD:HTR:140:BCTRL","QUAD:HTR:300:BCTRL"]
SCAN_VARIABLE = "QUAD:HTR:120:BCTRL"
QUAD_LENGTH = 0.1244 # m
DRIFT_LENGTH = 2.2 # m
BEAM_ENERGY = 0.06641461763347117 # GeV
PV_TO_INTEGRATED_GRADIENT = 1.0 # kG
ROI = None

SECONDARY_VARIABLES = pd.Series(["SOLN:GUNB:212:BCTRL",
                   "SOLN:GUNB:823:BCTRL",
                   "QUAD:GUNB:212:1:BCTRL",
                   "QUAD:GUNB:212:2:BCTRL",
                   "QUAD:GUNB:823:1:BCTRL",
                   "QUAD:GUNB:823:2:BCTRL",
                   "QUAD:HTR:120:BCTRL",
                   "QUAD:HTR:140:BCTRL",
                   "QUAD:HTR:300:BCTRL",
                   "QUAD:HTR:320:BCTRL",
                   "BEND:HTR:480:BACT"])

MEASUREMENT_OPTIONS = {
    "screen": SCREEN_NAME,
    "background": None,
    "roi": ROI,
    "bb_half_width": 3.0, # half width of the bounding box in terms of std
    "visualize": True,
}

IMAGE_CONSTRAINTS = {
    "bb_penalty": ["LESS_THAN", 0.0],
    "log10_total_intensity": ["GREATER_THAN", 4]
}

# define function to measure the total size on OTR4
def eval_beamsize(input_dict):
    results = measure_beamsize(input_dict)
    print(results)
    results["S_x_mm"] = np.array(results["Sx"]) * 1e-3
    results["S_y_mm"] = np.array(results["Sy"]) * 1e-3

    #add total beam size
    results["total_size"] = np.sqrt(np.array(results["Sx"])**2 + np.array(results["Sy"])**2)
    return results
