__author__ = "Karoline Stosio"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Karoline Stosio"]
__license__ = "MIT"
__maintainer__ = "Karoline Stosio"

import numpy as np
import glob
import cv2
import sys
import timeit
from scipy import ndimage
import gdk.config as config

#from ext.OnAVOS.segmenter import create_engine

#from gdk.imagproc.centroid_finder import center,get_objects_seg
#from gdk.imagproc.preprocessing import oclude_gripper

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

def get_bounding_box(mask,number=1):
	
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
		slice_x, slice_y, _ = ndimage.find_objects(label_im == number)[0]
	except:
		return None, None
	# roi = mask[slice_x, slice_y]

	new_slice_x, new_slice_y = make_square_box(slice_x, slice_y)
	# roi2 = mask[new_slice_x, new_slice_y]

	return new_slice_x, new_slice_y
	
def get_bounding_box_for_tracking(mask,number=1):   
	"""
	Extract coordinates in a format needed by opencv tracker- (x_tl, y_tl, w, h)
	"""
	sx,sy = get_bounding_box(mask,number)
	if sx is None:
		return False
	else:
		return (sy.start, sx.start, sy.stop-sy.start, sx.stop-sx.start)
   
def read_images_from_file(path):
	img_list = glob.glob(path+'/*')
	images = [cv2.imread(img) for img in sorted(img_list)]
	return images   

def get_separate_boxes(mask):
	"""
	Extracts bounding boxes from mask. Returns a list in which each 
	element is a bounding box in the opencv friendly format.
	"""
	if len(mask.shape)>2:
		mask = mask[:,:,0]
	ret, separate_masks = cv2.connectedComponents(mask)
	boxes = []
	for ii in range(1, np.max(separate_masks) + 1):
		bool_mask = (separate_masks == ii)
		curr_mask = mask * bool_mask
		curr_mask = np.stack((curr_mask,) * 3,2)
		bbox = get_bounding_box_for_tracking(curr_mask)
		if bbox: #sometimes no box is return
			boxes.append(bbox)
	print(boxes)
	return boxes

def draw_bounding_box(image,box,c=(200,0,0)):
	p1 = (int(box[0]), int(box[1]))
	p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
	cv2.rectangle(image, p1, p2, c)

def instantiate_trackers(mask,image,track_closest=False):
	trackers = []
	boxes = get_separate_boxes(mask)
	if track_closest:
		X,Y=config.XY_TRACK_POINT
		centroids = [get_center(bbox) for bbox in boxes]
		print(centroids)
		distances = [np.sqrt((c[0]-X)**2 + (c[1]-Y)**2) for c in centroids]
		print(distances)
		idx = np.argmin(distances)
		t = cv2.TrackerMedianFlow_create()
		ok = t.init(image, boxes[idx])
		trackers.append(t)
		print('chosen box {}'.format(boxes[idx]))
		return trackers,[boxes[idx]]
	else:
		for b in boxes:
			t = cv2.TrackerMedianFlow_create()
			ok = t.init(image, b)
			trackers.append(t)
			return trackers,boxes
	
def check_center(bbox,x,y):
	c1 = x+1 > bbox[0]
	c2 = x-1 < bbox[0]+bbox[2]
	c3 = y+1 > bbox[1]
	c4 = y-1 < bbox[1]+bbox[3]
	if c1 and c2 and c3 and c4:
		return True
	else:
		return False
	
def choose_closest_tracker(active_trackers,active_boxes,x,y):
	good_box_idx = np.where(np.array([check_center(bbox,x,y) for bbox in active_boxes])==1)[0][0]
	print(good_box_idx)
	return [active_trackers[good_box_idx]],[active_boxes[good_box_idx]]
	
def update_active_trackers(active_trackers,image):
	new_boxes = []
	bad_indices = []
	for i,t in enumerate(active_trackers):
		ok, newbox = t.update(image)
		if ok:
			new_boxes.append(newbox)
		else:
			bad_indices.append(i)
	for i in bad_indices[::-1]: #iterating in reverse to delete items without affecting indices
		t = active_trackers.pop(i)
		t.clear()
	if len(active_trackers)<1:
		print('All trackers lost!')
	return active_trackers, new_boxes
		
def update_trackers_with_new_segmentation(active_boxes,new_boxes,image,active_trackers):
	matrix = compare_all_trackers(active_boxes,new_boxes)
	indices_good_active_trackers = list(np.where(np.max(matrix,axis=0)==1)[0])
	indices_good_new_boxes = list(np.where(np.max(matrix,axis=1)==0)[0])
	new_active_trackers = []
	for i,tracker in enumerate(active_trackers):
		if i in indices_good_active_trackers:
			new_active_trackers.append(tracker)
		else:
			tracker.clear()
	for i in indices_good_new_boxes:
		t = cv2.TrackerMedianFlow_create()
		ok = t.init(image, new_boxes[i])
		new_active_trackers.append(t)
	return new_active_trackers# return, new_active_boxes as well
	
def compare_all_trackers(active_trackers,new_boxes):
	results = np.empty((len(new_boxes),len(active_trackers)),dtype=bool)
	for i,box in enumerate(new_boxes):
		for j,b in enumerate(active_trackers):
			results[i,j]=compare_boxes(b, box)
	return results

def compare_boxes(b1,b2):
	margin = 15
	c1 = np.abs(b1[0]-b2[0])<margin
	c2 = np.abs(b1[1]-b2[1])<margin
	c3 = np.abs(b1[0]+b1[2]-(b2[0]+b2[2]))<margin
	c4 = np.abs(b1[1]+b1[3]-(b2[1]+b2[3]))<margin
	if c1 and c2 and c3 and c4:
		return True
	else:
		return False

def get_img_from_bbox(image,bbox):
	img = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
	return img

def get_center(bbox):
	return (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2))

if __name__ == '__main__' :
 
	input_images = '/home/dlrc/Documents/Ach7-first_data/Raw_ims'
	
	engine = create_engine()
	
	images = read_images_from_file(input_images)
	size = (images[0].shape[0],images[0].shape[1])
	
	mask, prob = engine.segment(oclude_gripper(images[1]), size)
	active_trackers, new_boxes = instantiate_trackers(mask,images[1])
	
	# Iterate over frames 
	for i,image in enumerate(images[1:]):
		print(i)
		
		if i%5==0:
			mask, prob = engine.segment(oclude_gripper(image), size)
			segmentation_boxes = get_separate_boxes(mask)
			for bb in segmentation_boxes:
				draw_bounding_box(image,bb,c=(0,0,220))
			active_trackers=update_trackers_with_new_segmentation(new_boxes,segmentation_boxes,image,active_trackers)
		
		active_trackers, new_boxes = update_active_trackers(active_trackers,image)
		for b in new_boxes:
			draw_bounding_box(image,b)
		centroids = [get_center(bbox) for bbox in new_boxes]
		for xy in centroids:
			cv2.circle(frame, xy, 10, (255, 0, 0), 1)
		# Display all updated trackers
		cv2.imshow("Tracking", image)
	 
		# Exit if ESC pressed
		k = cv2.waitKey(0) 
		if k == 27 : break
