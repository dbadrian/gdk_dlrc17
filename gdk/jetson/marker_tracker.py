__author__ = "Akshat Tandon"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Akshat Tandon"]
__license__ = "MIT"
__maintainer__ = "Akshat Tandon"

import time
import sys
import logging
# Import PS-Drone
import cv2
import numpy as np
import gdk.config as config

import logging
import time
import os
import cv2
from cv2 import aruco
import numpy as np


import sys
sys.path.append('/usr/local/opencv')
from gdk.jetson.sensors import IMU, OnBoardCamera
from matplotlib import pyplot as plt
import cv2
import copy
import math



logger = logging.getLogger(__name__)


class MarkerTracker():

    def __init__(self):
    	self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    	self.arucoParams = aruco.DetectorParameters_create()
    	self.arucoParams.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
    	self.tracking = False
    	self.marker_ids=None

    def update(self, frame):
        self.marker_ids, self.corners = self.__get_corners_from_marker(frame)
        
        if self.marker_ids is not None:
            self.centroid = self.__get_centroid_from_corners()
            self.outer_corners = self.__get_main_corners_from_corners()
            self.height, self.width = frame.shape[:2]
        else:
            self.marker_ids = []

        return self.marker_ids

    def get_centroid_error(self, marker_id, state):
        if self.marker_ids is None:
            return None
        if marker_id in self.marker_ids:
            if state == "first_cam":
                errx = (self.centroid[marker_id][0] - config.XY_TRACK_POINT_MARKER[0])#/(config.XY_TRACK_POINT[0])
                erry = (self.centroid[marker_id][1] - config.XY_TRACK_POINT_MARKER[1])#/(config.XY_TRACK_POINT[1])
            elif state == "second_cam":
                errx = (self.centroid[marker_id][0] - config.XY_TRACK_POINT_MARKER_2[0])#/(config.XY_TRACK_POINT[0])
                erry = (self.centroid[marker_id][1] - config.XY_TRACK_POINT_MARKER_2[1])#/(config.XY_TRACK_POINT[1])
            return errx, erry
    
    def get_distance_error(self, marker_id):
        if self.marker_ids is None:
            return None
        if marker_id in self.marker_ids:
        	
        	marker_outer_corners=self.outer_corners[marker_id]
        	short_1 = np.linalg.norm(marker_outer_corners[0]-marker_outer_corners[1])
        	short_2 = np.linalg.norm(marker_outer_corners[3]-marker_outer_corners[2])
        	long_1 = np.linalg.norm(marker_outer_corners[1]-marker_outer_corners[3])
        	long_2 = np.linalg.norm(marker_outer_corners[2]-marker_outer_corners[0])
        	avg_short = (short_1+short_2)/2.0
        	avg_long = (long_1+long_2)/2.0
        	dif_short = (
                avg_short - config.BEST_DISTANCE[0])/config.BEST_DISTANCE[0]
        	dif_long = (avg_long - config.BEST_DISTANCE[1])/config.BEST_DISTANCE[1]

        	return (dif_short+dif_long)/2.0

    def __get_main_corners_from_corners(self):
        outer_corners=dict()
        if self.marker_ids is None:
            return None
        for i, mid in enumerate(self.marker_ids):
            outer_corners[int(mid)]=self.corners[i][0]
        return outer_corners

    def __get_centroid_from_corners(self):
        centroids=dict()
        if self.marker_ids is None:
            return None
        for i, mid in enumerate(self.marker_ids):

            
            centroids[int(mid)]=np.sum(np.array(self.corners[i][0]), 0) / float(len(np.array(self.corners[i][0])))
        return centroids

    def __get_corners_from_marker(self, frame):
        image = self.__preprocessing(frame)
        corners = None
        found=False
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.arucoParams)
        
        if ids is not None:
            #corners = np.array(corners)
            self.marker_ids = list(ids) 
            found=True
       
        return ids, corners

    def __preprocessing(self, img):
        logger.debug("Preprocessing image")
        #img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
#        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.GaussianBlur(img,(5,5),0)
        imgRemapped_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgRemapped_gray = cv2.equalizeHist(imgRemapped_gray)

        return imgRemapped_gray