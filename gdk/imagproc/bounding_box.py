__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

from scipy import ndimage
import numpy as np

def make_square_box(sx, sy):
    # calculate sizes of the box
    xx = sx.stop - sx.start
    yy = sy.stop - sy.start
    # the function asumes xx>yy, so the code below reverses variables if the case is opposite
    if yy > xx:
        (xx, yy, sy, sx) = (yy, xx, sx, sy)
        ref = 480
        big = 'y'
    else:
        ref = 640
        big = 'x'
    # check the padding
    if (xx - yy) % 2 == 0:
        new_y = (sy.start - (xx - yy) / 2, sy.stop + (xx - yy) / 2)
    if (xx - yy) % 2 == 1:
        new_y = y_slice = (sy.start - (xx - yy - 1) / 2, sy.stop + (xx - yy - 1) / 2 + 1)
    # check the borders
    if new_y[0] <= 0:
        new_y = (0, new_y[1] + abs(new_y[0]))
    elif new_y[1] >= ref:
        new_y = (new_y[0] - (new_y[1] - ref), ref)
    # return new slices
    if big == 'x':
        return sx, slice(int(new_y[0]), int(new_y[1]), None)
    if big == 'y':
        return slice(int(new_y[0]), int(new_y[1]), None), sx

def get_bounding_box(mask):
    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # Now that we have only one connect component, extract it's bounding box
    try:
        slice_x, slice_y, _ = ndimage.find_objects(label_im == 1)[0]
    except:
        return None,None
    # roi = mask[slice_x, slice_y]

    new_slice_x, new_slice_y = make_square_box(slice_x, slice_y)
    # roi2 = mask[new_slice_x, new_slice_y]

    return new_slice_x, new_slice_y