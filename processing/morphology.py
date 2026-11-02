import numpy as np

def erosion(image):
    padded = np.pad(image, pad_width=1, mode="constant", constant_values=0)

    result = (
        (padded[:-2, :-2] == 255) &
        (padded[:-2, 1:-1] == 255) &
        (padded[:-2, 2:] == 255) &
        (padded[1:-1, :-2] == 255) &
        (padded[1:-1, 1:-1] == 255) &
        (padded[1:-1, 2:] == 255) &
        (padded[2:, :-2] == 255) &
        (padded[2:, 1:-1] == 255) &
        (padded[2:, 2:] == 255)
    )

    return result.astype(np.uint8) * 255


def dilation(image):
    padded = np.pad(image, pad_width=1, mode="constant", constant_values=0)

    result = (
        (padded[:-2, :-2] == 255) |
        (padded[:-2, 1:-1] == 255) |
        (padded[:-2, 2:] == 255) |
        (padded[1:-1, :-2] == 255) |
        (padded[1:-1, 1:-1] == 255) |
        (padded[1:-1, 2:] == 255) |
        (padded[2:, :-2] == 255) |
        (padded[2:, 1:-1] == 255) |
        (padded[2:, 2:] == 255)
    )

    return result.astype(np.uint8) * 255
#Phep mo
def opening(image):
    return dilation(erosion(image))

#phep dong
def closing(image):
    return erosion(dilation(image))