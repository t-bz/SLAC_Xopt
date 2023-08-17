from scripts.utils.image_processing import get_beam_data
from scripts.evaluate_function.screen_image import measure_beamsize

import numpy as np
import matplotlib.pyplot as plt

img = np.load("../scripts/utils/test_img.npy")

get_beam_data(
    img,
    np.array([0, 0, 120, 110]),
    3000,
    visualize=True
)


plt.figure()
plt.imshow(img)
plt.show()
