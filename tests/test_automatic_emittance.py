import os

import matplotlib.pyplot as plt

import numpy as np

from scripts.automatic_emittance import BaseEmittanceMeasurement, BeamlineConfig
from scripts.characterize_emittance import characterize_emittance
from xopt import VOCS


def test_beamsize_function(inputs):
    x = inputs["x"]
    sx = np.sqrt(x**2 + x + 1) * (1.0 + np.random.randn(10) * 0.05)
    sy = np.sqrt(3 * x**2 + 2 * x + 1) * (1.0 + np.random.randn(10) * 0.05)

    if np.random.rand() > 0.5:
        sx[3] = np.NAN

    total_size = np.sqrt(np.array(sx) ** 2 + np.array(sy) ** 2)

    return {"Sx": sx, "Sy": sy, "total_size": total_size}


class TAutomaticEmittanceMeasurement(BaseEmittanceMeasurement):
    def eval_beamsize(self, inputs):
        results = test_beamsize_function(inputs)
        results["S_x_mm"] = results["Sx"]
        results["S_y_mm"] = results["Sy"]
        return results

    @property
    def measurement_vocs(self):
        vocs = VOCS(
            variables={
                self.beamline_config.scan_quad_pv: self.beamline_config.scan_quad_range
            },
            objectives={"total_size": "MINIMIZE"},
            observables=["S_x_mm", "S_y_mm"],
        )

        return vocs

    def get_initial_points(self):
        return self.measurement_vocs.random_inputs(3)


class TestAutomaticEmittance:
    def test_characterize_emittance(self):
        vocs = VOCS(
            variables={"x": [-5, 5]},
            objectives={"total_size": "MINIMIZE"},
            observables=["Sx", "Sy"],
        )

        beamline_config = BeamlineConfig(
            scan_quad_pv="x",
            scan_quad_range=vocs.variables["x"],
            scan_quad_length=0.1,
            transport_matrix_x=[[1.0, 1.0], [0.0, 1.0]],
            transport_matrix_y=[[1.0, 1.0], [0.0, 1.0]],
            beam_energy=1.0,
        )

        initial_data = vocs.random_inputs(1)

        print(initial_data)
        emit_result, emit_x = characterize_emittance(
            vocs,
            test_beamsize_function,
            beamline_config,
            "x",
            "Sx",
            "Sy",
            initial_data,
            quad_scan_analysis_kwargs={"visualize": True},
            n_iterations=10,
        )

        emit_x.data.plot(y="x")
        ax = emit_x.data.plot.scatter(x="x", y="total_size")
        tr = emit_x.generator.turbo_controller.get_trust_region(
            emit_x.generator.model
        ).flatten()
        print(emit_x.generator.turbo_controller)
        for ele in tr:
            ax.axvline(ele)

        # emit_x.data.to_csv("test_data.csv")

        plt.show()

    def test_automatic_emittance(self):
        beamline_config = BeamlineConfig(
            scan_quad_pv="x",
            scan_quad_range=[-5, 5],
            scan_quad_length=0.1,
            transport_matrix_x=[[1.0, 1.0], [0.0, 1.0]],
            transport_matrix_y=[[1.0, 1.0], [0.0, 1.0]],
            beam_energy=1.0,
        )

        emittance_measurement = TAutomaticEmittanceMeasurement(
            beamline_config=beamline_config,
            n_iterations=10,
            turbo_length=0.75,
            visualize=True,
        )

        results, emit_x = emittance_measurement.run()

        emit_x.data.plot(y="x")
        ax = emit_x.data.plot.scatter(x="x", y="total_size")
        tr = emit_x.generator.turbo_controller.get_trust_region(
            emit_x.generator.model
        ).flatten()
        for ele in tr:
            ax.axvline(ele)

        # emit_x.data.to_csv("test_data.csv")

        os.remove(emittance_measurement.dump_file)
        plt.show()
