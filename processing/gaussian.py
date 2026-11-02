import numpy as np

KERNEL = np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]]
    ,dtype=np.float32)/16



def gaussian_blur(image):
    padded = np.pad(image, pad_width=1, mode="edge").astype(np.float32)

    result = (
        padded[:-2, :-2] * KERNEL[0, 0]
        + padded[:-2, 1:-1] * KERNEL[0, 1]
        + padded[:-2, 2:] * KERNEL[0, 2]
        + padded[1:-1, :-2] * KERNEL[1, 0]
        + padded[1:-1, 1:-1] * KERNEL[1, 1]
        + padded[1:-1, 2:] * KERNEL[1, 2]
        + padded[2:, :-2] * KERNEL[2, 0]
        + padded[2:, 1:-1] * KERNEL[2, 1]
        + padded[2:, 2:] * KERNEL[2, 2]
    )

    return np.clip(result, 0, 255).astype(np.uint8)
