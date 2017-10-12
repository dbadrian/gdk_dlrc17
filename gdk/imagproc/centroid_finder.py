__author__ = "Akshat Tandon"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Akshat Tandon"]
__license__ = "MIT"

import cv2
import numpy as np

def center(mask, size):
    contours = get_objects_seg(mask)
    if contours:
        h, w = mask.shape[:2]
        cxl, cyl = find_centers(contours, h, w)
        cx, cy = find_closest(cxl, cyl, size)
    else:
        cx = None
        cy = None
    return cx, cy

def get_objects_seg(imgray):

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_centers(contours, h, w):
    img = np.zeros((h, w), np.uint8)
    cxl = []
    cyl = []

    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cxl.append(cX)
        cyl.append(cY)

        return cxl, cyl

def find_closest(cxl, cyl, size):
    wx, wy = size
    xa = np.asarray(cxl)
    ya = np.asarray(cyl)
    x2 = np.power(xa - wx, 2)
    y2 = np.power(ya - wy, 2)
    eucl_dist = np.sqrt(x2 + y2)
    min_idx = np.argmin(eucl_dist)
    return cxl[min_idx], cyl[min_idx]

