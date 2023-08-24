import time

import epics
import numpy as np
from edef import EventDefinition


# do measurements


def get_edef_data():
    PVS = ["GDET:FEE1:241:ENRC"]
    n_shots = 100
    try:
        with EventDefinition("roussel_edef", user="rroussel") as my_edef:
            my_edef.n_measurements = n_shots
            my_edef.beamcode = 0
            my_edef.start()
            while not my_edef.is_acquisition_complete():
                time.sleep(0.1)

            data = my_edef.get_data_buffer(PVS)

    except TypeError:
        print("tout")
        data = get_edef_data()

    return data


def do_measurement(inputs):
    """use pyepics to set input pvs, wait a few seconds and then do a measurement"""

    mean_charge = 2.61e2  # gun charge in pC
    charge_dev = 0.1  # factional charge deviation

    # set values
    for name, val in inputs.items():
        epics.caput(name, val)

    # wait
    time.sleep(10.0)

    # get 120 Hz data
    data = get_edef_data()

    # print(data)
    # calculate 80th percentile for SXR
    x = data["GDET:FEE1:241:ENRC"]
    x = x[~np.isnan(x)]
    data["GDET:FEE1:241:ENRC"] = np.percentile(x, 80.0)

    # calculate total losses
    soft_cblm_indexes = range(26, 48)
    hard_cblm_indexes = range(13, 46)
    soft_loss_PVS = [f"CBLM:UNDS:{ele}10:I1_LOSS" for ele in soft_cblm_indexes]
    hard_loss_PVS = [f"CBLM:UNDH:{ele}75:I1_LOSS" for ele in hard_cblm_indexes]

    data["TMITH"] = epics.caget("BPMS:LI30:201:TMITCUH1H") / 1e9
    data["TMITS"] = epics.caget("BPMS:LI30:201:TMITCUS1H") / 1e9

    losses = epics.caget_many(soft_loss_PVS + hard_loss_PVS)

    data["TOTAL_SOFT_LOSSES"] = np.sum(losses[: len(soft_loss_PVS)])
    data["TOTAL_HARD_LOSSES"] = np.sum(losses[len(soft_loss_PVS) + 1 :])

    # get averaged pulse intensity for HXR
    data["EM2K0:XGMD:HPS:AvgPulseIntensity"] = epics.caget(
        "EM2K0:XGMD:HPS:AvgPulseIntensity"
    )

    data["time"] = time.time()
    data["DUMMY"] = 1.0

    return data
