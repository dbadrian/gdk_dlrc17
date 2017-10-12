__author__ = "Karoline Stosio"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Karoline Stosio, Lucia Seitz"]
__license__ = "MIT"
__maintainer__ = "Karoline Stosio"

import numpy as np
import glob
import cv2
import sys
import timeit
from scipy import ndimage

from ext.OnAVOS.segmenter import create_engine

from gdk.imagproc.centroid_finder import center,get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper
from gdk.imagproc.multitracking import *

if __name__ == '__main__' :

	input_images = '/home/dlrc/Documents/Ach7-first_data/Raw_ims'

	engine = create_engine()

	images = read_images_from_file(input_images)
	size = (images[0].shape[0],images[0].shape[1])
	mask, prob = engine.segment(oclude_gripper(images[0]), size)
	#cv2.imshow("Mask", mask)

	active_trackers, new_boxes = instantiate_trackers(mask,images[0],track_closest=True)

	brick_images=[get_img_from_bbox(images[0],new_boxes[0])]


	# Iterate over frames
	for i,image in enumerate(images):
		print(i)
		'''
		if i%5==0:
			print('Im here')
			mask, prob = engine.segment(oclude_gripper(image), size)

			segmentation_boxes = get_separate_boxes(mask)
			for bb in segmentation_boxes:
				draw_bounding_box(image,bb,c=(0,0,220))

			active_trackers = update_trackers_with_new_segmentation(new_boxes,segmentation_boxes,image,active_trackers)
		'''
		active_trackers, new_boxes = update_active_trackers(active_trackers,image)
		if len(new_boxes)>0:
			brick_images.append(get_img_from_bbox(image,new_boxes[0]))
		for b in new_boxes:
			draw_bounding_box(image,b)
		centroids = [get_center(bbox) for bbox in new_boxes]
		for xy in centroids:
			print(xy)
			cv2.circle(image, xy, 10, (255, 0, 0), 1)

		'''
		good_xy = (396, 125)
		print(new_boxes)
		if not HAVE_GOAL:
			active_trackers,new_boxes=choose_closest_tracker(active_trackers,new_boxes,good_xy[0],good_xy[1])
			HAVE_GOAL = True
		'''
		# Display all updated trackers
		#cv2.imshow("Tracking", image)
		cv2.imshow("Tracking", oclude_gripper(image))
		cv2.imshow("Mask", mask)
		cv2.imshow("Brick", brick_images[-1])
		# Exit if ESC pressed
		k = cv2.waitKey(0)
		if k == 27 : break
