import pandas as pd
import warnings

import time

import numpy as np
import yaml
from epics import PV


def get_pv_objects(filename):
    tracked_pvs = yaml.safe_load(open(filename))
    return {ele: PV(ele) for ele in tracked_pvs}


def measure_pvs(names, pv_objects):
    return {name: pv_objects[name].get() for name in names}


def load_data(fname):
    config = yaml.safe_load(open(fname))
    data = pd.DataFrame(config["data"])
    data.index = map(int, data.index)
    data = data.sort_index(axis=0)
    return data


def save_reference_point(pv_dict: dict[str, PV], filename="reference.yml"):
    """save a set of reference values that can be used to return the machine to a
    previous state"""

    values = {name: ele.get() for name, ele in pv_dict.items()}
    yaml.dump(values, open(filename, "w"))

    return values


def set_magnet_strengths(
    strengths_dict: dict[str, float], pv_dict: dict[str, PV], validate=True
):
    """set magnet strengths using epics pvs, wait for BACT to readback same value"""
    for bctrl_name, val in strengths_dict.items():
        bact_name = bctrl_name.replace("BCTRL", "BACT")

        # set using BCTRL
        print(bctrl_name, val)
        pv_dict[bctrl_name].put(val)

        # relative and absolute tolerances for setting pvs
        rtol = 1e-3
        atol = 1e-6

        # wait until magnet read back matches set point, timeout = 1000 cycles (100 sec)
        if validate:
            i = 0
            while ~np.isclose(val, pv_dict[bact_name].get(), rtol=rtol, atol=atol):
                time.sleep(0.1)
                i += 1

                if i > 1000:
                    warnings.warn(
                        f"timeout exceeded while waiting for {bact_name} to "
                        f"reach setpoint, skipping"
                    )
                    break
