__author__ = "Jonathon Luiten"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Jonathon Luiten, David Adrian"]
__license__ = "MIT"
__maintainer__ = "Jonathon Luiten"

import time
import json
import logging
import argparse
import os

import cv2
import ev3dev.ev3 as ev3
import numpy as np
from scipy import ndimage

import gdk.jetson.broker as Broker
# import gdk.jetson.sensors as Sensors
import gdk.jetson.controller as Controller
import gdk.jetson.tracker as Tracker
import gdk.config as config
import gdk.common as common
import gdk.utils as utils
import gdk.statemachine.statemachine as Statemachine

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center,get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper

# Currently up here, cause OnAVOSold causes conflicts with the logger
common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--jetson', '-j', action='store_true',
                    help="This parameter should be set if run from a Jetson board with internal camera")
parser.add_argument('--expected_slaves', '-es', required=True, nargs='+', type=str,
                    help="The (list) of clients which need to connect to the server before processing can start.")
parser.add_argument('--interface', '-i', required=True, type=str,
                    help="Interface to which ev3 is connect.")
args = parser.parse_args()

if not utils.master_announcement(args.expected_slaves, args.interface,
                                 message_delay=5, repeats=2):
    pass
    #exit()
# else:
config.BROKER_IP = utils.find_interface_ip(args.interface)

# sleep shortly so every service can get ready...ye sucks
#time.sleep(2)

# Start the MMQT Messaging system
logger.info("MMQT Broker: Initalization")
broker = Broker.Broker(
    config.BROKER_IP, config.BROKER_PORT, broker_data={})
logger.info("Broker: Connecting")
broker.connect()

controller = Controller.RobotController(broker)
robot = EV3Robot(controller, jetson_mode=args.jetson)
statemachine = Statemachine.RobotBrain("Ben")

logger.info("Initializing Robot")
robot.startup()
