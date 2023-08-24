import os

import numpy as np
import yaml

from scripts.image import ImageDiagnostic
from scripts.utils.read_files import read_file


class TestImageDiagnostic:
    def test_load_from_file(self):
        ImageDiagnostic.parse_obj(yaml.safe_load(open("TEST_config.yml")))

    def test_statistics(self):
        diagnostic = ImageDiagnostic.parse_obj(yaml.safe_load(open("TEST_config.yml")))
        diagnostic.return_statistics = True

        result = diagnostic.measure_beamsize(5)
        assert np.isclose(result["Cx_var"], 0.0)
        assert np.isclose(result["Sx_var"], 0.0)
        assert isinstance(result["Sx"], float)

    def test_image_saving(self):
        diagnostic = ImageDiagnostic.parse_obj(yaml.safe_load(open("TEST_config.yml")))

        # set save image location
        diagnostic.save_image_location = os.getcwd()

        result = diagnostic.measure_beamsize(3)

        # read file
        file_info = read_file(result["save_filename"])
        assert file_info["resolution"] == 1.0

        os.remove(result["save_filename"])

    def test_fitting_fail(self):
        class BadImageDiagnostic(ImageDiagnostic):
            def fit_image(self, img):
                # simulate a failure to fit one axis
                para_x = [np.NaN] * 4
                para_y = [1.0] * 4

                return {
                    "centroid": np.array((para_x[1], para_y[1])),
                    "rms_sizes": np.array((para_x[2], para_y[2])),
                    "total_intensity": img.sum(),
                    "log10_total_intensity": np.log10(img.sum()),
                }

        bad_image_diagnostic = BadImageDiagnostic(screen_name="TEST")
        result = bad_image_diagnostic.calculate_beamsize(np.ones((20, 20)))
        assert result["Cx"] == np.Nan
