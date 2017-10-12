__author__ = "Lucia Seitz"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Lucia Seitz"]
__license__ = "MIT"
__maintainer__ = "Lucia Seitz"

import os
import time
import glob
import cv2
from scipy import ndimage
import numpy as np
from ext.OnAVOS.segmenter import create_engine
from ext.OnAVOS.LatentSpace import create_latent_engine
from gdk.network_interface import image_to_latents

from gdk.imagproc.centroid_finder import center,get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper
from gdk.imagproc.bounding_box import get_bounding_box

def read_images_from_file(path):
    img_list = glob.glob(path+'/*')
    images = [cv2.imread(img) for img in img_list]
    return images

def check_one_brick(mask):
    contours = get_objects_seg(mask)
    #_,separate_masks = cv.connectedComponents(mask)
    return len(contours)==1

segment_engine = create_engine()
latent_engine = create_latent_engine()
input_images = '/home/dlrc/lucia/acc34/videos/images_brick1'

folder = "/home/dlrc/Karo/out/"
im_fol = "vid_images/"
mask_fol = "vid_masks/"
data_fol = "DATA/"
all_im_fol = "all_im/"


# if not os.path.exists(folder + im_fol):
# 	os.makedirs(folder + im_fol)
# if not os.path.exists(folder + mask_fol):
# 	os.makedirs(folder + mask_fol)
# if not os.path.exists(folder + data_fol):
# 	os.makedirs(folder + data_fol)
# if not os.path.exists(folder + all_im_fol):
# 	os.makedirs(folder + all_im_fol)

# images = read_images_from_file(input_images)
# size = (int(images[0].shape[0]),int(images[0].shape[1]))
# good_examples = []
#
# for i,image in enumerate(images):
#     print(i)
#     name = str(i).zfill(4)
#     mask, prob = segment_engine.segment(oclude_gripper(image), size)
#
#     # cv2.imwrite(folder + all_im_fol + name + '.png', cv2.resize(mask, (200, 200)))
#     # cv2.imwrite(folder + all_im_fol + name + '.jpg', cv2.resize(image, (200, 200)))
#
#     print('masked')
#     if check_one_brick(mask):
#         print('1 brick')
#         mask = np.stack((mask,) * 3,2)
#         bool_mask = mask > mask.mean()
#         # masked_im = image * bool_mask
#         masked_im = image
#         slice_x, slice_y = get_bounding_box(mask)
#         small_im = masked_im[slice_x, slice_y]
#         small_im = cv2.resize(small_im, (64, 64))
#         good_examples.append(small_im)
#
#         print('saving')
#
#         cv2.imwrite(folder + data_fol + name + '.png', small_im)
#
#     # cv2.imwrite(folder + im_fol + name + '.jpg', image)
#     # mask = mask[:, :, 0]
#     # cv2.imwrite(folder + mask_fol + name + '.png', mask)
#     # cv2.imwrite(folder + data_fol + name + '.png', small_im)

############
# CODE HERE IS WRONG

# images = read_images_from_file(input_images)
# size = (int(images[0].shape[0]),int(images[0].shape[1]))
# good_examples = []
#
# latents = []
# for i,image in enumerate(images):
#     print(i)
#     now_latents, coords, small_ims, elapsed = image_to_latents(image, segment_engine, latent_engine, scale=1, occlude=True,mask_image=False)
#     name = str(i).zfill(4)
#     if len(small_ims) == 1:
#         good_examples.append(small_ims[0])
#     latents.append(now_latents[0])


############



good_examples = [cv2.imread(folder+data_fol+name, cv2.IMREAD_COLOR) for name in sorted(os.listdir(folder+data_fol))]

latents = [latent_engine.get_latent(im) for im in good_examples]

###########

mean = np.mean(latents,axis=0)
var = np.var(latents,axis=0)
dists = [np.sqrt(np.sum((mean - current_latent)**2)) for current_latent in latents]
max_dist = np.max(dists,axis = 0)
print (mean)
print (var)
print(max_dist)
print(dists)


fol_base = '/home/dlrc/lucia/acc34/videos/images_angles1/angle'

for i in range(8):
    fol = fol_base + str(i)+"/"
    fn1 = os.listdir(fol)[0]
    print (fol+fn1)
    image = cv2.imread(fol + fn1, cv2.IMREAD_COLOR)
    latents,coords,small_ims,elapsed = image_to_latents(image,segment_engine,latent_engine,scale=0.5,occlude=False,mask_image=False)
    dists = [np.sqrt(np.sum((mean - current_latent)**2)) for current_latent in latents]
    print ("angle" + str(i))
    print(dists)
    print(coords)
    print(elapsed)

    col_no = (0, 0, 255)
    col_yes = (0, 255, 0)

    for (dist,coord) in zip(dists,coords):
        if dist<max_dist:
            col = col_yes
        else:
            col = col_no
        cv2.circle(image, coord, 10,col, 4)
        cv2.putText(image, str(dist), (coord[0]+20,coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col)

    cv2.putText(image, "max=" + str(max_dist), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_yes)
    cv2.imshow('tracked_point', image)
    cv2.waitKey(0)








