import math
from PIL import Image
import math
RESOLUTIONS_17 = [
    (672, 1568),(688,1504),(720,1456),(752,1392),(800,1328),
    (832,1248),(880,1184),(944,1104),(1024,1024),(1104,944),
    (1184,880),(1248,832),(1328,800),(1392,752),(1456,720),
    (1504,688),(1568,672),
]
RATIO = {
    'any_17ratio': [
        (w // math.gcd(w, h), h // math.gcd(w, h))
        for w, h in RESOLUTIONS_17
    ],
    'any_11ratio': [(16, 9), (9, 16), (7, 5), (5, 7), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_9ratio': [(16, 9), (9, 16), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_7ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_5ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (1, 1)],
    'any_1ratio': [(1, 1)],
}

def pick_ratio(orig_h: int, orig_w: int, anyres: str = 'any_17ratio') -> tuple[int,int]:

    orig_ratio = orig_w / orig_h
    rw, rh = min(
        RATIO[anyres],
        key=lambda pair: abs((pair[0]/pair[1]) - orig_ratio)
    )
    return rw, rh


def compute_size(
    rw: int, rh: int, stride: int,
    *, min_pixels=None, max_pixels=None, anchor_pixels=None
) -> tuple[int,int]:

    base_w, base_h = rw * stride, rh * stride
    area = base_w * base_h


    if anchor_pixels is not None:
        target_area = anchor_pixels
    elif min_pixels is not None and max_pixels is not None:

        if area > max_pixels:
            target_area = max_pixels
        elif area < min_pixels:
            target_area = min_pixels
        else:
            target_area = area
    else:
        target_area = area

    scale = math.sqrt(target_area / area)

    new_w = max(stride, int(base_w * scale)) // stride * stride
    new_h = max(stride, int(base_h * scale)) // stride * stride
    return new_h, new_w


def dynamic_resize(
    orig_h: int, orig_w: int,
    anyres: str = 'any_17ratio',
    anchor_pixels: int = 1024*1024,
    stride: int = 32
) -> tuple[int,int]:


    rw, rh = pick_ratio(orig_h, orig_w, anyres)


    base_w, base_h = rw * stride, rh * stride
    area = base_w * base_h
    s = max(1, round(math.sqrt(anchor_pixels / area)))

    new_w = (base_w * s) // stride * stride
    new_h = (base_h * s) // stride * stride
    return new_h, new_w


def concat_images_adaptive(images, bg_color=(255, 255, 255)):

    if not images:
        raise ValueError("images is empty")

    n = len(images)
    

    cols = int(n**0.5)
    if cols * cols < n:
        cols += 1
    rows = (n + cols - 1) // cols


    widths, heights = zip(*(img.size for img in images))
    max_w = max(widths)
    max_h = max(heights)


    new_img = Image.new('RGB', (cols * max_w, rows * max_h), color=bg_color)

    for idx, img in enumerate(images):
        row_idx = idx // cols
        col_idx = idx % cols

        offset_x = col_idx * max_w + (max_w - img.width) // 2
        offset_y = row_idx * max_h + (max_h - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))

    return new_img