# {@article{Miskovich:2022kqg,
#     author = "Miskovich, Sara and Edelen, Auralee and Mayes, Christopher",
#     title = "{PyEmittance: A General Python Package for Particle Beam Emittance Measurements with Adaptive Quadrupole Scans}",
#     doi = "10.18429/JACoW-IPAC2022-TUPOST059",
#     journal = "JACoW",
#     volume = "IPAC2022",
#     pages = "TUPOST059",
#     year = "2022"
# }

import numpy as np
from scripts.utils.fitting_methods import (
    fit_gaussian_linear_background,
)

import logging

logger = logging.getLogger(__name__)


class Image:
    """Beam image processing and fitting for beamsize, amplitude, centroid"""

    def __init__(self, image, nrow, ncol, bg_image=None):
        self.nrow = nrow
        self.ncol = ncol
        self.flat_image = image
        self.bg_image = bg_image
        self.offset = 20

        self.proc_image = None
        self.x_proj = None
        self.y_proj = None
        self.xrms = None
        self.yrms = None
        self.xrms_error = None
        self.yrms_error = None
        self.xcen = None
        self.ycen = None
        self.xcen_error = None
        self.ycen_error = None
        self.xamp = None
        self.yamp = None
        self.xamp_error = None
        self.yamp_error = None

    def reshape_im(self, im=None):
        """Reshapes flattened OTR image to 2D array"""

        self.proc_image = self.flat_image.reshape(self.nrow, self.ncol)
        return self.proc_image

    def subtract_bg(self):
        """Subtracts bg image"""

        if self.bg_image is not None:

            if self.bg_image.endswith(".npy"):
                self.bg_image = np.load(self.bg_image)
            else:
                logger.info("Error in load bg_image: not .npy format.")
                return self.proc_image

            self.bg_image = self.bg_image.reshape(self.nrow, self.ncol)
            if self.proc_image.shape == self.bg_image.shape:
                self.proc_image = self.proc_image - self.bg_image
                # some pixels may end up with negative data
                # set element in image that are <0 to 0
                self.proc_image = np.array(
                    [e if e >= 0 else 0 for ele in self.proc_image for e in ele]
                )
                self.proc_image = self.proc_image.reshape(self.nrow, self.ncol)
            else:
                logger.info("Beam image and background image are not the same shape.")

        return self.proc_image

    def get_im_projection(self, subtract_baseline=True):
        """Expects ndarray, return x (axis=0) or y (axis=1) projection"""

        self.x_proj = np.sum(self.proc_image, axis=0)
        self.y_proj = np.sum(self.proc_image, axis=1)

        if subtract_baseline:
            self.x_proj = self.x_proj - np.mean(self.x_proj[0: self.offset])
            self.y_proj = self.y_proj - np.mean(self.y_proj[0: self.offset])

        return self.x_proj, self.y_proj

    def get_sizes(self, show_plots=True, n_restarts=50):
        """Takes an image (2D array) and optional bg image, finds x and y projections,
        and fits with desired method. Current options are "gaussian" or "rms cut area".
        Returns size in x, size in y, error on x size, error on  y size"""

        # Find statistics
        para_x = fit_gaussian_linear_background(
            self.x_proj, inital_guess=None, show_plots=show_plots, n_restarts=n_restarts
        )
        para_y = fit_gaussian_linear_background(
            self.y_proj, inital_guess=None, show_plots=show_plots, n_restarts=n_restarts
        )

        self.xamp, self.yamp = (
            para_x[0],
            para_y[0],
            #para_error_x[0],
            #para_error_y[0],
        )

        self.xcen, self.ycen = (
            para_x[1],
            para_y[1],
            #para_error_x[1],
            #para_error_y[1],
        )

        #      size in x, size in y, error on x size, error on  y size
        self.xrms, self.yrms = (
            para_x[2],
            para_y[2],
            #para_error_x[2],
            #para_error_y[2],
        )

        return {
            "rms_sizes": np.array((self.xrms, self.yrms)),
            "rms_error": np.array((self.xrms_error, self.yrms_error)),
            "centroid": np.array((self.xcen, self.ycen)),
            "total_intensity": self.proc_image.sum()
        }

