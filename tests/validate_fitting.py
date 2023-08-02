import os

import numpy as np

from scripts.utils.image_processing import get_beam_data
from pyemittance.image import Image

data_dir = "D:/SLAC/LCLS/Injector/emit_characterize_1/"

files = os.listdir(data_dir)

for name in files[:50:5]:
    img = np.load(data_dir + name).astype(np.double)
    im = Image(img.flatten(), *img.shape)
    im.reshape_im()

    profx, profy = im.get_im_projection()

    print(get_beam_data(im.proc_image, visualize=True))

    fit_res = im.get_sizes(method = "gaussian", show_plots = False)
    #xsize, ysize, xsize_error, ysize_error, x_amplitude, y_amplitude = fit_res
    print(fit_res)
    print(im.xcen, im.ycen)