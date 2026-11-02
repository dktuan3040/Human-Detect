import numpy as np
def merge_near_boxes(boxes, gap=40):
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        result = []
        while boxes:
            x1, y1, w1, h1 = boxes.pop()
            i = 0
            while i < len(boxes):
                x2, y2, w2, h2 = boxes[i]
                # hai ô gần/chồng nhau?
                if (x1 - gap < x2 + w2 and x2 - gap < x1 + w1 and
                    y1 - gap < y2 + h2 and y2 - gap < y1 + h1):
                    nx, ny = min(x1, x2), min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    x1, y1, w1, h1 = nx, ny, nw, nh
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            result.append([x1, y1, w1, h1])
        boxes = result
    return [tuple(b) for b in boxes]
def find_connected_components(image):
    height, width = image.shape

    visited = np.zeros_like(image, dtype=bool)
    components = []
    runs = []
    uf = UnionFind()
    for y in range(height):
        row = image[y]

        white_pixels = np.where(row == 255)[0]

        if len(white_pixels) == 0:
            continue

        splits = np.where(np.diff(white_pixels) > 1)[0] + 1
        groups = np.split(white_pixels, splits)

        for group in groups:
            x_start = group[0]
            x_end = group[-1]

            label = uf.make_set()

            runs.append((y, x_start, x_end, label))

    for i in range(len(runs)):
        y1, x1_start, x1_end, label1 = runs[i]

        for j in range(i + 1, len(runs)):
            y2, x2_start, x2_end, label2 = runs[j]

            if y2 > y1 + 1:
                break

            if y2 == y1 + 1:
                if x1_start <= x2_end + 1 and x1_end + 1 >= x2_start:
                    uf.union(label1, label2)        

    groups = {}

    for y, x_start, x_end, label in runs:
        root = uf.find(label)

        if root not in groups:
            groups[root] = {
                "x_min": x_start,
                "x_max": x_end,
                "y_min": y,
                "y_max": y
            }
        else:
            groups[root]["x_min"] = min(groups[root]["x_min"], x_start)
            groups[root]["x_max"] = max(groups[root]["x_max"], x_end)
            groups[root]["y_min"] = min(groups[root]["y_min"], y)
            groups[root]["y_max"] = max(groups[root]["y_max"], y)           

    for group in groups.values():
        x = group["x_min"]
        y = group["y_min"]

        w = group["x_max"] - x + 1
        h = group["y_max"] - y + 1

        components.append((x, y, w, h))             
    return components

    merged = boxes.copy()

    changed = True

    while changed:
        changed = False

        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):

                if is_near(merged[i], merged[j], margin):
                    new_box = merge_boxes(merged[i], merged[j])

                    merged[i] = new_box
                    merged.pop(j)

                    changed = True
                    break

            if changed:
                break

    return merged
class UnionFind:
    def __init__(self):
        self.parent = []

    def make_set(self):
        index = len(self.parent)
        self.parent.append(index)
        return index

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a