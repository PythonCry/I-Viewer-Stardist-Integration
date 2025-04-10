import time
import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO


def map_coords(r, patch_info):
    # trim border objects, map to original coords
    # patch_info: [x0_s, y0_s, w_p(w_s), h_p(h_s), pad_w(x0_p), pad_h(y0_p)]
    x0_s, y0_s, w_p, h_p, x0_p, y0_p = patch_info
    # assert x0_p == 64 and y0_p == 64 and w_s == w_p and h_s == h_p, f"{roi_slide}, {roi_patch}"
    x_c, y_c = r['boxes'][:,[0,2]].mean(1), r['boxes'][:,[1,3]].mean(1)
    keep = (x_c > x0_p) & (x_c < x0_p + w_p) & (y_c > y0_p) & (y_c < y0_p + h_p)

    res = {k: r[k][keep] for k in ['boxes', 'labels', 'scores']}
    res['boxes'][:, [0, 2]] += x0_s - x0_p
    res['boxes'][:, [1, 3]] += y0_s - y0_p
    if 'masks' in r:
        res['masks'] = [m + [x0_s - x0_p, y0_s - y0_p] 
                        for m, tag in zip(r['masks'], keep) if tag]

    return res


def export_detections_to_table(res, labels_text=None, save_masks=True):
    df = {}
    df['x0'], df['y0'], df['x1'], df['y1'] = np.round(res['boxes']).astype(np.int32).T  
    if labels_text is not None:
        df['label'] = [labels_text[x] for x in res['labels'].tolist()]
    else:
        df['label'] = [f'cls_{x}' for x in res['labels']]
    if 'scores' in res:
        df['score'] = res['scores'].round(decimals=4)
    if save_masks and 'masks' in res:
        poly_x, poly_y = [], []
        for poly in res['masks']:
            poly_x.append(','.join([f'{_:.2f}' for _ in poly[:, 0]]))
            poly_y.append(','.join([f'{_:.2f}' for _ in poly[:, 1]]))

        df['poly_x'] = poly_x
        df['poly_y'] = poly_y

    return pd.DataFrame(df)


def pad_pil(img, pad_width, color=0):
    pad_l, pad_r, pad_u, pad_d = pad_width
    w, h = img.size

    res = Image.new(img.mode, (w + pad_l + pad_r, h + pad_u + pad_d), color)
    res.paste(img, (pad_l, pad_u))

    return res


def get_dzi(image_size, tile_size=254, overlap=1, format='jpeg'):
    """ Return a string containing the XML metadata for the .dzi file.
        image_size: (w, h)
        tile_size: tile size
        overlap: overlap size
        format: the format of the individual tiles ('png' or 'jpeg')
    """
    import xml.etree.ElementTree as ET
    image = ET.Element(
        'Image',
        TileSize=str(tile_size),
        Overlap=str(overlap),
        Format=format,
        xmlns='http://schemas.microsoft.com/deepzoom/2008',
    )
    w, h = image_size
    ET.SubElement(image, 'Size', Width=str(w), Height=str(h))
    tree = ET.ElementTree(element=image)
    buf = BytesIO()
    tree.write(buf, encoding='UTF-8')

    return buf.getvalue().decode('UTF-8')
