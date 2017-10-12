import time
import sys
import logging
# Import PS-Drone
import cv2
import numpy as np
import gdk.config as config

logger = logging.getLogger(__name__)


class CheckerBoardTracker():

    def __init__(self):
        self.tracking = False

    def update(self, frame):
        self.tracking, self.corners = self.__get_corners_from_marker(frame)

        if self.tracking:
            self.centroid = self.__get_centroid_from_corners()
            self.outer_corners = self.__get_main_corners_from_corners()
            self.height, self.width = frame.shape[:2]

        return self.tracking

    def get_centroid_error(self):
        if self.tracking:
            errx = (self.centroid[0][0] - config.XY_TRACK_POINT[0])#/(config.XY_TRACK_POINT[0])
            erry = (self.centroid[0][1] - config.XY_TRACK_POINT[1])#/(config.XY_TRACK_POINT[1])
            return errx, erry
    
    def get_distance_error(self):
        if self.tracking:
            short_1 = np.linalg.norm(self.outer_corners[0]-self.outer_corners[1])
            short_2 = np.linalg.norm(self.outer_corners[3]-self.outer_corners[2])
            long_1 = np.linalg.norm(self.outer_corners[1]-self.outer_corners[3])
            long_2 = np.linalg.norm(self.outer_corners[2]-self.outer_corners[0])

            avg_short = (short_1+short_2)/2.0
            avg_long = (long_1+long_2)/2.0

            dif_short = (
                avg_short - config.BEST_DISTANCE[0])/config.BEST_DISTANCE[0]
            dif_long = (avg_long - config.BEST_DISTANCE[1])/config.BEST_DISTANCE[1]

            return (dif_short+dif_long)/2.0

    def __get_main_corners_from_corners(self):
        return np.array([self.corners[0][0], self.corners[3][0], self.corners[16][0], self.corners[19][0]])

    def __get_centroid_from_corners(self):
        return np.sum(self.corners, 0) / float(len(self.corners))

    def __get_corners_from_marker(self, frame):
        corners = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, config.PATTERN_SIZE, corners, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)
        npcorners = np.array(corners)
        return found, npcorners

        

