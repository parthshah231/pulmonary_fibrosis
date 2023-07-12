from typing import List, Tuple, Dict

import scipy
import matplotlib.pyplot as plt
import numpy as np

from skimage import measure
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from sklearn.cluster import KMeans

from pydicom import FileDataset


def best_rect(m: int) -> Tuple[int, int]:
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Not possible!")


def resample_instances(instances: List[FileDataset], new_spacing=[1, 1, 1]):
    try:
        spacing = np.array(
            [instances[0].SliceThickness] + list(instances[0].PixelSpacing)
        )
    except AttributeError:
        raise ValueError("DICOM file is missing required attributes")

    imgs = np.array([instance.pixel_array for instance in instances])

    resize_factor = spacing / new_spacing
    new_shape = np.round(imgs.shape * resize_factor)
    rounded_resize_factor = new_shape / imgs.shape
    rounded_new_spacing = spacing / rounded_resize_factor

    imgs = scipy.ndimage.zoom(imgs, rounded_resize_factor, mode="nearest")

    return imgs, rounded_new_spacing


def perform_windowing(instance: FileDataset, bounds: Tuple[int, int]):
    min_bound, max_bound = bounds
    img = instance.pixel_array
    img[img < min_bound] = min_bound
    img[img > max_bound] = max_bound
    return img


def make_mask(instance: FileDataset, display: bool = False):
    img = instance.pixel_array

    row_size = img.shape[0]
    col_size = img.shape[1]

    # Normalize the image
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    # Grab the center (usually helps in removing unnecessary black borders)
    middle = img[
        int(col_size / 5) : int(col_size / 5 * 4),
        int(row_size / 5) : int(row_size / 5 * 4),
    ]

    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum
    # underflow - anything below the mean minus the standard deviation
    # overflow - anything above the mean plus the standard deviation

    img[img == max] = mean
    img[img == min] = mean

    # Using K-means to separate foreground (soft tissue / bone) and background (air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilated = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(
        dilated
    )  # Different labels are displayed in different colors
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if (
            B[2] - B[0] < row_size / 10 * 9
            and B[3] - B[1] < col_size / 10 * 9
            and B[0] > row_size / 5
            and B[2] < col_size / 5 * 4
        ):
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([8, 8]))  # one last dilation

    if display:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("Original CT-scan")
        ax[0].imshow(img, cmap="gray")
        ax[1].set_title("Pixel array")
        ax[1].imshow(mask, cmap="gray")
        plt.show()

    return mask
