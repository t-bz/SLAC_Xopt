import json

import yaml

from lcls_ii.emittance.bax.emittance_opt import BAXEmittanceOpt
from scripts.image import ImageDiagnostic, ROI


class TestBAXEmittanceOpt:
    def tests(self):
        data_dir = "."
        pv_list = "../../tracked_pvs.yml"

        fname = "otr_diagnostic.yml"  # run_dir + "OTRS_HTR_330_config.yml"

        roi = ROI(ycenter=967, xcenter=878, xwidth=600, ywidth=600)

        image_diagnostic = ImageDiagnostic.parse_obj(yaml.safe_load(open(fname)))
        image_diagnostic.roi = roi
        image_diagnostic.min_log_intensity = 5.0
        image_diagnostic.save_image_location = data_dir
        image_diagnostic.n_fitting_restarts = 2
        image_diagnostic.visualize = False

        tuning_variables = {
            "QUAD:HTR:120:BCTRL": [-4.0, 1.0],
            "QUAD:HTR:140:BCTRL": [-4.4778, 4.4762],
            "QUAD:HTR:300:BCTRL": [-4.46035, 4.4692],
        }
        measurement_variable = {"QUAD:HTR:320:BCTRL": [-4.46919, 4.4792]}
        QUAD_LENGTH = 0.124  # m
        rmat_x = [[1.0, 0.2481], [0.0, 1.0]]
        rmat_y = [[1.0, 0.2481], [0.0, 1.0]]
        BEAM_ENERGY = 0.088  # GeV
        SCALE_FACTOR = 2.74
        x_design_twiss = [5.011, 0.0487]
        y_design_twiss = [5.011, 0.0487]

        bax_emit_opt = BAXEmittanceOpt(
            data_dir=data_dir,
            pv_list=pv_list,
            image_diagnostic=image_diagnostic,
            tuning_variables=tuning_variables,
            measurement_variable=measurement_variable,
            quad_length=QUAD_LENGTH,
            rmat_x=rmat_x,
            rmat_y=rmat_y,
            beam_energy_gev=BEAM_ENERGY,
            pv_to_geometric_focusing_strength_scale=SCALE_FACTOR,
            x_design_twiss=x_design_twiss,
            y_design_twiss=y_design_twiss,
        )

        yaml.dump(
            json.loads(bax_emit_opt.model_dump_json()), open("test_dump.yml", "w")
        )

        # test loading
        load_config = yaml.safe_load(open("test_dump.yml"))
        load_bax = BAXEmittanceOpt(**load_config)

        # test getting tune/meas dims
        tuning_dims, measurement_dim = load_bax.get_tune_meas_dims(load_bax.get_vocs())
        assert tuning_dims == [0, 1, 2]
        assert measurement_dim == 3

        # test getting vocs
        vocs = load_bax.get_vocs()
        assert vocs.constraints == {"bb_penalty": ["LESS_THAN", 0.0]}
        assert vocs.variables == {
            "QUAD:HTR:120:BCTRL": [-4.0, 1.0],
            "QUAD:HTR:140:BCTRL": [-4.4778, 4.4762],
            "QUAD:HTR:300:BCTRL": [-4.46035, 4.4692],
            "QUAD:HTR:320:BCTRL": [-4.46919, 4.4792],
        }

        # test getting model constructor
        load_bax.get_model_construtor()
