from scripts.evaluate_function.screen_image import measure_beamsize
from scripts.utils.read_files import read_file


class TestMeasureBeamSize:
    def test_saving(self):
        test_inputs = {
            "screen": "test_img.npy",
            "save_img_location": ".",
            "n_shots": 5,
            "visualize": False,
            "dummy_pv": "dummy",
            "sleep_time": 0.01
        }
        results = measure_beamsize(test_inputs)
        saved_results = read_file(results["save_filename"])
        assert saved_results["images"].shape == (5, 120, 192)
        assert saved_results["dummy_pv"] == "dummy"
        assert len(saved_results["Cx"]) == 5

        import os
        os.remove(results["save_filename"])
