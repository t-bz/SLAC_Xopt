import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scripts.utils.py_emittance_image_processing import Image


def get_beam_data(
        img: np.ndarray,
        roi_data: np.ndarray = None,
        bb_half_width: float = 2.0,
        min_log_intensity: float = None,
        visualize: bool = True
):
    """
        A method for processing raw screen images with a beam.

        As part of the analysis this function adds a bounding box (BB) around the beam
        distribution. The maximum BB distance from ROI cente is usable as a
        constraint, referred here as a `penalty` value. If less than zero, BB is
        entirely inside circular ROI, otherwise it is outside ROI. If the penalty
        function is positive, centroid and rms values returned are Nans.

        Returns a dict containing the following elements
            "Cx": beam centroid location in x
            "Cy": beam centroid location in y
            "Sx": rms beam size in x
            "Sy": rms beam size in y
            "penalty": penalty function value

        a region of interest (ROI) is specified as
        :                +------------------+
        :                |                  |
        :              height               |
        :                |                  |
        :               (xy)---- width -----+

        Parameters
        ----------
        img : np.ndarray
            n x m image data
        roi_data : np.ndarray, optional
            list containing roi bounding box elements [x, y, width, height]
        threshold: float, optional
            value to subtract from raw image, negative values after subtraction are
            set to zero
        bb_half_width: float, optional
            Bounding box half width in terms of standard deviations
        min_log_intensity: float, optional
            minimum log intensity that sets results to Nans
        visualize: bool, default: False
            flag to plot image and bounding box after processing

        Returns
        -------
        results : dict
            results dict
        """

    min_log_intensity = min_log_intensity or 0.0

    x_size, y_size = img.shape

    if roi_data is not None:
        if (roi_data[0] + roi_data[2]) > x_size or (roi_data[1] + roi_data[3]) > y_size:
            raise ValueError(f"must specify ROI that is smaller than the image, "
                             f"image size is {img.shape}")

        img = img[
              roi_data[0]:roi_data[0] + roi_data[2],
              roi_data[1]:roi_data[1] + roi_data[3]
              ]

    roi_c = np.array(img.shape) / 2
    roi_radius = np.min((roi_c * 2, np.array(img.shape))) / 2

    img_obj = Image(img.flatten(), *img.shape)
    img_obj.reshape_im()

    img_obj.get_im_projection()

    fit = img_obj.get_sizes(show_plots=visualize)
    centroid = fit["centroid"]
    sizes = fit["rms_sizes"]
    total_intensity = fit["total_intensity"]

    # get beam region
    n_stds = bb_half_width
    pts = np.array(
        (
            centroid - n_stds * sizes,
            centroid + n_stds * sizes,
            centroid - n_stds * sizes * np.array((-1, 1)),
            centroid + n_stds * sizes * np.array((-1, 1))
        )
    )

    #print(pts)
    #print(c)
    
    # visualization
    if visualize:
        fig, ax = plt.subplots()
        c = ax.imshow(img, origin="lower")
        ax.plot(*centroid, "+r")
        ax.plot(*roi_c[::-1], ".r")
        fig.colorbar(c)

        rect = patches.Rectangle(pts[0], *sizes * n_stds * 2.0, facecolor='none',
                                 edgecolor="r")
        ax.add_patch(rect)

        circle = patches.Circle(roi_c[::-1], roi_radius, facecolor="none",
                                edgecolor="r")
        ax.add_patch(circle)

        #plt.figure()
        #plt.plot(img.sum(axis=0))

        #plt.figure()
        #plt.plot(img.sum(axis=1))

    distances = np.linalg.norm(pts - roi_c, axis=1)

    # subtract radius to get penalty value
    bb_penalty = np.max(distances) - roi_radius
    log10_total_intensity = np.log10(total_intensity)

    results = {
        "Cx": centroid[0],
        "Cy": centroid[1],
        "Sx": sizes[0],
        "Sy": sizes[1],
        "bb_penalty": bb_penalty,
        "total_intensity": total_intensity,
        "log10_total_intensity": log10_total_intensity
    }

    # set results to none if the beam extends beyond the roi or if the intensity is
    # not greater than a minimum
    if bb_penalty > 0 or log10_total_intensity < min_log_intensity:
        for name in ["Cx", "Cy", "Sx", "Sy"]:
            results[name] = np.NaN

    # set bb penalty to None if there is no beam
    if log10_total_intensity < min_log_intensity:
        results["bb_penalty"] = np.NaN

    return results


def calculate_stats(img):
    rows, cols = img.shape
    row_coords = np.arange(rows)
    col_coords = np.arange(cols)

    m00 = np.sum(img)
    m10 = np.sum(col_coords[:, np.newaxis] * img.T)
    m01 = np.sum(row_coords[:, np.newaxis] * img)

    Cx = m10 / m00
    Cy = m01 / m00

    m20 = np.sum((col_coords[:, np.newaxis] - Cx) ** 2 * img.T)
    m02 = np.sum((row_coords[:, np.newaxis] - Cy) ** 2 * img)

    sx = (m20 / m00) ** 0.5
    sy = (m02 / m00) ** 0.5

    return Cx, Cy, sx, sy
