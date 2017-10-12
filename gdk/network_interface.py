__author__ = "Jonothan Luiten"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Jonothan Luiten, David Adrian"]
__license__ = "MIT"
__maintainer__ = "Jonothan Luiten"

import time
import cv2
import numpy as np

from gdk.imagproc.centroid_finder import center
from gdk.imagproc.preprocessing import oclude_gripper
from gdk.imagproc.bounding_box import get_bounding_box
import gdk.utils as utils

def net_segment_image(image, segment_engine, latent_engine, scale = 1,occlude = False, mask_image = False, compute_latent=False, final_latent = False):
    start = time.time()
    size = (int(image.shape[1]*scale), int(image.shape[0]*scale))

    latents = []
    coords = []
    boxes = []
    small_ims = []

    if final_latent:
        small_im = image
        small_im = cv2.resize(small_im, (64, 64))
        small_ims.append(small_im)
        latents.append(utils.encode_numpy_array(latent_engine.get_latent(small_im)))
        mask = np.empty([size[0],size[1],3])

        import os
        fol = "/home/dlrc/aaaa/"
        num_fol = len(os.listdir(fol))-1
        if not os.path.exists(fol + str(num_fol).zfill(4)):
            os.makedirs(fol + str(num_fol).zfill(4))
        num_im = len(os.listdir(fol + str(num_fol).zfill(4)))
        cv2.imwrite(fol + str(num_fol).zfill(4) + "/" + str(num_im).zfill(4) + "_test.jpg", small_im)

    else:

        # if occlude:
        #     image = oclude_gripper(image)

        resized_image = cv2.resize(image, size)
        resize_mask, resized_prob = segment_engine.segment(resized_image, (size[1],size[0]))
        mask = cv2.resize(resize_mask, (image.shape[1], image.shape[0]))

        if occlude:
            mask = oclude_gripper(mask)

        ret, seperate_masks = cv2.connectedComponents(mask)

        for ii in range(1, seperate_masks.max().max() + 1):
            bool_mask = (seperate_masks == ii)
            curr_mask = mask * bool_mask

            x, y = center(curr_mask, size)

            curr_mask = np.stack((curr_mask,) * 3, 2)

            if mask_image:
                bool_mask = curr_mask > curr_mask.mean()
                masked_im = image * bool_mask
            else:
                masked_im = image

            slice_x, slice_y = get_bounding_box(curr_mask)
            if slice_x is None:
                continue

            small_im = masked_im[slice_x, slice_y]
            small_im = cv2.resize(small_im, (64, 64))
            boxes.append((slice_y.start, slice_x.start, slice_y.stop-slice_y.start, slice_x.stop-slice_x.start))

            if compute_latent:
                latents.append(utils.encode_numpy_array(latent_engine.get_latent(small_im)))
                small_ims.append(small_im)

            coords.append((x,y))

        print(compute_latent)
        if compute_latent:
            import os
            fol = "/home/dlrc/aaaa/"
            num_fol = len(os.listdir(fol))
            if not os.path.exists(fol + str(num_fol).zfill(4)):
                os.makedirs(fol + str(num_fol).zfill(4))
            cv2.imwrite(fol + str(num_fol).zfill(4) + "/orig.jpg", image)
            cv2.imwrite(fol + str(num_fol).zfill(4) + "/mask.jpg", mask)
            print(fol + str(num_fol).zfill(4) + "/orig.jpg")
            print(fol + str(num_fol).zfill(4) + "/mask.jpg")
            for i, small_im in enumerate(small_ims):
                cv2.imwrite(fol + str(num_fol).zfill(4) + "/" + str(i).zfill(4) + "_small.jpg", small_im)
                print(fol + str(num_fol).zfill(4) + "/" + str(i).zfill(4) + "_small.jpg")

    end = time.time()
    elapsed = end - start

    return latents, coords, utils.encode_numpy_array(mask), boxes, small_ims

def net_segment_image_batch(image, segment_engine, latent_engine, scale = 1,occlude = False, mask_image = False, compute_latent=False):
    start = time.time()
    size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    if occlude:
        image = oclude_gripper(image)

    resized_image = cv2.resize(image, size)
    resize_mask, resized_prob = segment_engine.segment(resized_image, (size[1],size[0]))
    mask = cv2.resize(resize_mask, (image.shape[1], image.shape[0]))

    ret, seperate_masks = cv2.connectedComponents(mask)
    latents = []
    coords = []
    small_ims = []
    boxes = []

    for ii in range(1, seperate_masks.max().max() + 1):
        bool_mask = (seperate_masks == ii)
        curr_mask = mask * bool_mask

        x, y = center(curr_mask, size)


        curr_mask = np.stack((curr_mask,) * 3, 2)

        if mask_image:
            bool_mask = curr_mask > curr_mask.mean()
            masked_im = image * bool_mask
        else:
            masked_im = image

        slice_x, slice_y = get_bounding_box(curr_mask)
        if slice_x is None:
            continue

        small_im = masked_im[slice_x, slice_y]
        small_im = cv2.resize(small_im, (64, 64))
        small_ims.append(small_im)

        boxes.append((slice_y.start, slice_x.start, slice_y.stop-slice_y.start, slice_x.stop-slice_x.start))
        coords.append((x,y))

    if compute_latent:
        latents_temp = latent_engine.get_latent_batch(small_ims)
        for jj in range(latents_temp.shape[0]):
            a = np.expand_dims(latents_temp[jj], 0)
            if a is np.ndarray:
                latents.append(utils.encode_numpy_array(a))

    end = time.time()
    elapsed = end - start

    return latents, coords, utils.encode_numpy_array(mask), boxes


def image_to_latents(image, segment_engine, latent_engine, scale = 1,occlude = False, mask_image = False):
    start = time.time()
    size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    if occlude:
        image = oclude_gripper(image)

    resized_image = cv2.resize(image, size)
    resize_mask, resized_prob = segment_engine.segment(resized_image, (size[1],size[0]))
    mask = cv2.resize(resize_mask, (image.shape[1], image.shape[0]))

    ret, seperate_masks = cv2.connectedComponents(mask)
    latents = []
    coords = []
    small_ims = []

    for ii in range(1, seperate_masks.max().max() + 1):
        bool_mask = (seperate_masks == ii)
        curr_mask = mask * bool_mask

        x, y = center(curr_mask, size)

        curr_mask = np.stack((curr_mask,) * 3, 2)

        if mask_image:
            bool_mask = curr_mask > curr_mask.mean()
            masked_im = image * bool_mask
        else:
            masked_im = image

        slice_x, slice_y = get_bounding_box(curr_mask)
        if slice_x is None:
            continue

        small_im = masked_im[slice_x, slice_y]
        small_im = cv2.resize(small_im, (64, 64))

        latents.append(latent_engine.get_latent(small_im))
        coords.append((x,y))
        small_ims.append(small_im)

    end = time.time()
    elapsed = end - start

    return latents,coords,small_ims,mask,elapsed

def image_to_latents_batch(image, segment_engine, latent_engine, scale = 1,occlude = False, mask_image = False):
    start = time.time()
    size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    if occlude:
        image = oclude_gripper(image)

    resized_image = cv2.resize(image, size)
    resize_mask, resized_prob = segment_engine.segment(resized_image, (size[1],size[0]))
    mask = cv2.resize(resize_mask, (image.shape[1], image.shape[0]))

    ret, seperate_masks = cv2.connectedComponents(mask)
    latents = []
    coords = []
    small_ims = []

    for ii in range(1, seperate_masks.max().max() + 1):
        bool_mask = (seperate_masks == ii)
        curr_mask = mask * bool_mask

        x, y = center(curr_mask, size)

        curr_mask = np.stack((curr_mask,) * 3, 2)

        if mask_image:
            bool_mask = curr_mask > curr_mask.mean()
            masked_im = image * bool_mask
        else:
            masked_im = image

        slice_x, slice_y = get_bounding_box(curr_mask)
        if slice_x is None:
            continue

        small_im = masked_im[slice_x, slice_y]
        small_im = cv2.resize(small_im, (64, 64))

        # latents.append(latent_engine.get_latent(small_im))
        coords.append((x,y))
        small_ims.append(small_im)

    latents = latent_engine.get_latent_batch(small_ims)
    # latents = np.zeros([seperate_masks.max().max() + 1,32])
    end = time.time()
    elapsed = end - start

    return latents,coords,small_ims,mask,elapsed


def image_to_latents_small(image, segment_engine, latent_engine, scale = 1,occlude = False, mask_image = False):
    # occlude = True
    start = time.time()
    size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    if occlude:
        image = oclude_gripper(image)

    resized_image = cv2.resize(image, size)
    resize_mask, resized_prob = segment_engine.segment(resized_image, (size[1],size[0]))
    mask = cv2.resize(resize_mask, (image.shape[1], image.shape[0]))

    # cv2.imshow("1", mask)
    #
    kernel = np.ones((19, 19), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    #
    # cv2.imshow("2", mask)

    new_mask = np.zeros((image.shape[0],image.shape[1]))
    ret, seperate_masks = cv2.connectedComponents(mask)
    for ii in range(1, seperate_masks.max().max() + 1):
        bool_mask = (seperate_masks == ii)
        curr_mask = mask * bool_mask
        nnz = np.count_nonzero(curr_mask)
        if(nnz<200):
            continue
        new_mask += curr_mask
    mask = new_mask.astype("uint8")

    mask[mask<200] = 0
    mask[mask>=200] = 255

    # cv2.imshow("3",mask)
    #
    # # kernel = np.ones((19, 19), np.uint8)
    # # mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    #
    # cv2.imshow("4", mask)
    # cv2.waitKey(0)

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    ret, seperate_masks = cv2.connectedComponents(mask)
    latents = []
    coords = []
    small_ims = []

    for ii in range(1, seperate_masks.max().max() + 1):
        bool_mask = (seperate_masks == ii)
        curr_mask = mask * bool_mask

        x, y = center(curr_mask, size)

        curr_mask = np.stack((curr_mask,) * 3, 2)

        if mask_image:
            bool_mask = curr_mask > curr_mask.mean()
            masked_im = image * bool_mask
        else:
            masked_im = image

        slice_x, slice_y = get_bounding_box(curr_mask)
        if slice_x is None:
            continue

        # New code: not yet tested - not sure if works - don't use
        # slice_x = slice(int(min(slice_x.start -5,0)),int(max(slice_x.stop + 5,image.shape[0])),None)
        # slice_y = slice(int(min(slice_y.start - 5, 0)), int(max(slice_y.stop + 5, image.shape[0])), None)
        # slice_x.stop = max(slice_x.stop + 5,image.shape[0])
        # slice_y.start = min(slice_y.start - 5, 0)
        # slice_y.stop = max(slice_y.stop + 5, image.shape[1])

        small_im = masked_im[slice_x, slice_y]
        small_im = cv2.resize(small_im, (64, 64))

        latents.append(latent_engine.get_latent(small_im))
        coords.append((x,y))
        small_ims.append(small_im)

    end = time.time()
    elapsed = end - start

    return latents,coords,small_ims,mask,elapsed