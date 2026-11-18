import numpy as np


def threshold_image(image, threshold_value):
    result = np.zeros(image.shape, dtype=np.uint8)

    result[image > threshold_value] = 255

    return result
