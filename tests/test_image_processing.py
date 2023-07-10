from scripts.utils.image_processing import get_beam_data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.load("test_img.npy")

get_beam_data(
    img,
    np.array([0, 0, 120, 110]),
    3000,
    visualize=True
)
plt.figure()
plt.imshow(img)
plt.show()
