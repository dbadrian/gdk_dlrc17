#!/usr/bin/env python
__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"
__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

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

# Currently up here, cause OnAVOSold causes conflicts with the logger
common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ext.OnAVOS.segmenter import create_engine

seg_q = Queue(maxsize=1)


def _receive_img(client, userdata, msg):
    # DECODE IMAGE
    # func, args = json.loads(msg.payload.decode())
    logger.debug("got image")
    data = json.loads(msg.payload.decode())
    seg_q.put((data))


def _subscribe_img_topic(client, userdata, flags, rc):
    logger.debug("Connected with result code "+str(rc))
    logger.debug("Subscribing to topic=seg_img")
    client.subscribe("seg_img")


def init_img_topic(client):
    client.on_message = _receive_img
    client.on_connect = _subscribe_img_topic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-viz', action='store_true',
                        help="Activate this to enable visualization output")
    parser.add_argument(
        '-p_rec', type=str, help='Path to store recorded images. If defined, it automatically actives saving.')
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
        # exit()
    # else:
    config.BROKER_IP = utils.find_interface_ip(args.interface)

    # Start the MMQT Messaging system
    logger.info("MMQT Broker: Initalization")
    broker = Broker.Broker(
        config.BROKER_IP, config.BROKER_PORT, broker_data={})
    logger.info("Broker: Connecting")
    broker.connect()
    init_img_topic(broker.client)

    if args.jetson:
        cam = Sensors.OnBoardCamera()
        cam.open()
    else:
        cam = Sensors.ExternalCamera(1)
        cam.open()

    sent_image = None
    most_recent_mask = None
    broker.client.loop_start()  # (max_packets=1000)

    while True:
        logger.debug("read camera frame")
        frame = cam.read()
        if not type(sent_image) is np.ndarray:
            logger.debug("Sending very first frame")
            sent_image = np.copy(frame)
            broker.net_segment_image(sent_image, compute_latent=False, scale=0.5)

        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)

        if seg_q.full():
            logger.debug("Got image in queue, sending new frame")
            most_recent_segmentation = seg_q.get()
            most_recent_image = np.copy(sent_image)
            sent_image = np.copy(frame)
            broker.net_segment_image(sent_image, compute_latent=False, scale=0.5)

            latents = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
            logger.info("new seg data \n%s\n%s", latents,most_recent_segmentation["coords"])
            # do everything here which requires updating onces a new mask was received


if __name__ == "__main__":
    main()
