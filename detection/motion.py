import numpy as np
from processing.threshold import threshold_image

def detect_motion(previous_frame, current_frame,threshold_value):
    difference = np.abs(
        current_frame.astype(np.int16) -
        previous_frame.astype(np.int16)
    )
    motion_mask = threshold_image(difference,threshold_value)
    return motion_mask

def calculate_motion_ratio(motion_mask):
    white_pixels = np.count_nonzero(motion_mask == 255)

    total_pixels = motion_mask.size

    ratio = white_pixels / total_pixels * 100

    return ratio