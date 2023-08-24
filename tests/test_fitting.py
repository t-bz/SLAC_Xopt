import os

import numpy as np

from scripts.utils.fitting_methods import fit_gaussian_linear_background
import matplotlib.pyplot as plt


class TestImageFitting:
    def test_image_fitting(self):
        data_dir = "D:/SLAC/LCLS/Injector/emit_characterize_1/"

        files = os.listdir(data_dir)

        for name in files[100:200:5]:
            img = np.load(data_dir + name).astype(np.double)

            x_projection = np.sum(img, axis=0)
            x_projection = x_projection - x_projection.min()
            fit_gaussian_linear_background(x_projection)

        plt.show()