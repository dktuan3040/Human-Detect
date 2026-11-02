import numpy as np


def threshold_image(image, threshold_value):
    result = np.zeros_like(image)

    result[image > threshold_value] = 255

    return result