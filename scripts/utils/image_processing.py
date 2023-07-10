import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_beam_data(
        img: np.ndarray,
        roi_data: np.ndarray,
        threshold: float,
        min_log_intensity: float = 5.5,
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
        roi_data : np.ndarray
            list containing roi bounding box elements [x, y, width, height]
        threshold: int
            value to subtract from raw image, negative values after subtraction are
            set to zero
        min_log_intensity: float, default: 5.5
            minimum log intensity that sets results to Nans
        visualize: bool, default: False
            flag to plot image and bounding box after processing

        Returns
        -------
        results : dict
            results dict
        """
    x_size, y_size = img.shape
    if (roi_data[0] + roi_data[2]) > x_size or (roi_data[1] + roi_data[3]) > y_size:
        raise ValueError(f"must specify ROI that is smaller than the image, "
                         f"image size is {img.shape}")

    cropped_image = img[
                    roi_data[0]:roi_data[0] + roi_data[2],
                    roi_data[1]:roi_data[1] + roi_data[3]
                    ]

    filtered_image = gaussian_filter(cropped_image, 3.0)

    thresholded_image = np.where(
        filtered_image - threshold > 0, filtered_image - threshold, 0
    )

    # get circular ROI region
    roi_c = np.array((roi_data[2], roi_data[3])) / 2
    roi_radius = np.min((roi_c * 2, np.array(thresholded_image.shape))) / 2

    # set intensity outside circular ROI to zero
    xidx = np.arange(cropped_image.shape[0])
    yidx = np.arange(cropped_image.shape[1])
    mesh = np.meshgrid(xidx, yidx)
    outside_roi = np.sqrt(
        (mesh[0] - roi_c[0]) ** 2 + (mesh[1] - roi_c[1]) ** 2
    ) > roi_radius
    thresholded_image[outside_roi.T] = 0

    total_intensity = np.sum(thresholded_image)

    cx, cy, sx, sy = calculate_stats(thresholded_image)
    c = np.array((cx, cy))
    s = np.array((sx, sy))

    # get beam region
    n_stds = 2
    pts = np.array(
        (
            c - n_stds * s,
            c + n_stds * s,
            c - n_stds * s * np.array((-1, 1)),
            c + n_stds * s * np.array((-1, 1))
        )
    )

    # visualization
    if visualize:
        fig, ax = plt.subplots()
        c = ax.imshow(thresholded_image, origin="lower")
        ax.plot(cx, cy, "+r")
        ax.plot(*roi_c[::-1], "+r")
        fig.colorbar(c)

        rect = patches.Rectangle(pts[0], *s * n_stds * 2.0, facecolor='none',
                                 edgecolor="r")
        ax.add_patch(rect)

        circle = patches.Circle(roi_c[::-1], roi_radius, facecolor="none", edgecolor="r")
        ax.add_patch(circle)

    distances = np.linalg.norm(pts - roi_c, axis=1)

    # subtract radius to get penalty value
    bb_penalty = np.max(distances) - roi_radius
    log10_total_intensity = np.log10(total_intensity)

    results = {
        "Cx": cx,
        "Cy": cy,
        "Sx": sx,
        "Sy": sy,
        "bb_penalty": bb_penalty,
        "total_intensity": total_intensity,
        "log10_total_intensity": log10_total_intensity
    }

    if bb_penalty > 0 or log10_total_intensity < min_log_intensity:
        for name in ["Cx", "Cy", "Sx", "Sy"]:
            results[name] = None

    if log10_total_intensity < min_log_intensity:
        results["bb_penalty"] = None

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
