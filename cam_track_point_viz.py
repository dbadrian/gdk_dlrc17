__author__ = "Akshat Tandon"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Akshat Tandon, Jonathon Luiten, David Adrian"]
__license__ = "MIT"


import time
import json
import logging
import argparse
from multiprocessing import Queue, Process

import cv2
import ev3dev.ev3 as ev3
import numpy as np

import gdk.jetson.broker as Broker
import gdk.jetson.sensors as Sensors
import gdk.jetson.controller as Controller
import gdk.jetson.tracker as Tracker
import gdk.config as config
import gdk.common as common
import gdk.utils as utils
import gdk.statemachine.statemachine as Statemachine

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center
from gdk.imagproc.preprocessing import oclude_gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jetson', '-j', action='store_true',
                        help="This parameter should be set if run from a Jetson board with internal camera")
    args = parser.parse_args()

    count = 0
    cam=None
    if args.jetson:
        cam = Sensors.OnBoardCamera()
        cam.open()
    else:
        cam = Sensors.ExternalCamera(1)
        cam.open()
    while True:
        image=cam.read()
        target_xy = config.XY_TRACK_POINT
        cv2.circle(image, target_xy, 10, (255, 0, 255), 1)
        cv2.imshow("Camera output:",image)
        kp = cv2.waitKey(5) & 0xFF

        if kp == ord("s"):
            cv2.imwrite("/tmp/cam_" +str(count) + ".png", image)
            count += 1
        elif kp == ord("c"):
            exit()

if __name__ == "__main__":
    main()
