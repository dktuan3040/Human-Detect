import numpy as np
from detection.motion import calculate_motion_ratio

PERSON_RATIO = 2   # pixel trắng của box phải chiếm > 2% cả khung hình
SHOULDER_RATIO = 1.4     # phần thân bắt đầu phải rộng >= 1.4 lần đầu
MIN_HEAD_RATIO = 0.15    # đầu phải rộng >= 15% bề ngang box
HEAD_HEIGHT_MIN = 0.1   # chiều cao đầu tối thiểu / chiều rộng đầu
HEAD_HEIGHT_MAX = 3.0   # chiều cao đầu tối đa / chiều rộng đầu
BODY_HEIGHT_RATIO = 0.3 # xét đoạn thân cao bằng x lần chiều cao đầu

GAP_MIN = 1        # khe trống >= x cột thì tách
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
    
    if crop.ndim != 2 or crop.size == 0:
        return False, 0.0
    rows = (crop > 0).sum(axis=1).astype(np.float32)
    occupied = np.flatnonzero(rows)
    if occupied.size < 6:
        return False, 0.0
    rows = rows[occupied[0]:occupied[-1] + 1]
    box_w = crop.shape[1]

    for body_start in range(3, len(rows)):
        head_h = body_start
        head_rows = rows[:body_start]
        # Phân vị 90 giảm ảnh hưởng của một hàng nhiễu rộng bất thường.
        head_w = float(np.percentile(head_rows, 90))
        if head_w < max(2, MIN_HEAD_RATIO * box_w):
            continue
        if not HEAD_HEIGHT_MIN <= head_h / head_w <= HEAD_HEIGHT_MAX:
            continue
        body_h = max(3, int(round(head_h * BODY_HEIGHT_RATIO)))
        body_end = body_start + body_h
        if body_end > len(rows):
            continue
        body_rows = rows[body_start:body_end]
        if np.any(rows[:body_end] == 0):
            continue
        # Gặp phần rộng liên tục thì dừng; không lấy max ở tận đáy box.
        if (rows[body_start] >= SHOULDER_RATIO * head_w and
                np.mean(body_rows >= SHOULDER_RATIO * head_w) >= 0.8):
            body_w = float(np.median(body_rows))
            return True, body_w / head_w
    return False, 0.0
def is_person(mask, box, frame_shape):
    x, y, w, h = box

    box_mask = np.zeros_like(mask)
    box_mask[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    ratio = calculate_motion_ratio(box_mask)

    ok_shape, body_ratio = has_head_and_shoulders(mask[y:y + h, x:x + w])

    print(f"ratio={ratio:.2f}% body/head={body_ratio:.2f}")
    return ratio > PERSON_RATIO and ok_shape
