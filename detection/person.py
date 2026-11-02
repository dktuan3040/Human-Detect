import numpy as np
from detection.motion import calculate_motion_ratio

PERSON_RATIO = 2   # pixel trắng của box phải chiếm > 2% cả khung hình
SHOULDER_RATIO = 1.4     # vai phải rộng >= 1.4 lần đầu
NECK_DROP = 0.85         # độ rộng tụt dưới 85% đỉnh đầu -> coi là cổ
MIN_HEAD_RATIO = 0.15    # đầu phải rộng >= 15% bề ngang box

GAP_MIN = 3        # khe trống >= 3 cột thì tách
EMPTY_MAX = 2      # cột có <= 2 pixel trắng coi là cột trống
MIN_PIECE_W = 10   # mảnh hẹp hơn 10 cột thì bỏ (nhiễu)

def split_by_gap(mask, box):
    x, y, w, h = box
    crop = mask[y:y + h, x:x + w]
    has_white = (crop > 0).sum(axis=0) > EMPTY_MAX

    pieces = []
    start = None
    last = None
    empty_run = 0
    for i, has in enumerate(has_white):
        if has:
            if start is None:
                start = i
            last = i
            empty_run = 0
        elif start is not None:
            empty_run += 1
            if empty_run >= GAP_MIN:
                pieces.append((start, last))
                start = None
                empty_run = 0
    if start is not None:
        pieces.append((start, last))

    result = []
    for c0, c1 in pieces:
        pw = c1 - c0 + 1
        if pw < MIN_PIECE_W:
            continue
        sub = crop[:, c0:c1 + 1]
        rows = np.nonzero((sub > 0).sum(axis=1))[0]
        if rows.size == 0:
            continue
        result.append((x + c0, y + rows[0], pw, rows[-1] - rows[0] + 1))

    return result or [box]
def has_head_and_shoulders(crop):
    rows = (crop > 0).sum(axis=1).astype(np.float32)
    rows = np.convolve(rows, np.ones(5) / 5, mode="same")
    box_w = crop.shape[1]

    head_w = 0
    neck_row = None
    for i, r in enumerate(rows):
        if r > head_w:
            head_w = r
        elif head_w >= MIN_HEAD_RATIO * box_w and r < NECK_DROP * head_w:
            neck_row = i
            break

    if neck_row is None:
        return False, 0

    shoulder_w = rows[neck_row:].max()
    ratio = shoulder_w / head_w
    return ratio >= SHOULDER_RATIO, ratio
def is_person(mask, box, frame_shape):
    x, y, w, h = box

    box_mask = np.zeros_like(mask)
    box_mask[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    ratio = calculate_motion_ratio(box_mask)

    ok_shape, shoulder = has_head_and_shoulders(mask[y:y + h, x:x + w])

    print(f"ratio={ratio:.2f}% shoulder={shoulder:.2f}")
    return ratio > PERSON_RATIO and ok_shape