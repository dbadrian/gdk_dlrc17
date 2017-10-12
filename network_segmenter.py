__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import logging
import json
import argparse
import time

import paho.mqtt.client as mqtt
import numpy as np

import cv2

import gdk.config as config
import gdk.utils as utils
import gdk.common as common

common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ext.OnAVOS.segmenter import create_engine
from gdk.network_interface import net_segment_image, net_segment_image_batch
from ext.OnAVOS.LatentSpace import create_latent_engine
from debugger import debugger
# from ext.OnAVOS.LatentSpace import create_small_latent_engine2 as create_latent_engine



def _subscribe_img_topic(client, userdata, flags, rc):
    logger.debug("Connected with result code "+str(rc))
    logger.debug("Subscribing to topic=img")
    client.subscribe("img")


def _receive_img(client, userdata, msg):
    t1 = time.time()
    func, args = json.loads(msg.payload.decode())
    image = utils.decode_numpy_array(args["image"])
    logger.debug("%s %s", image.shape, image.dtype)
    logger.debug("Doing Segmentation")
    latents, coords, mask, boxes, small_ims = net_segment_image(image, userdata["engine"], userdata["latent_engine"], occlude=args["occlude"], compute_latent=args["compute_latent"], final_latent=args["final_latent"], scale=args["scale"])
    logger.debug("Publishing Coords/Latents")
    logger.debug("got %d latents, and %d coords and %d boxes", len(latents), len(coords), len(boxes))
    data = {"latents": latents,
            "coords": coords,
            "boxes": boxes
    }
    if "return_mask" in args and args["return_mask"]:
        logger.debug("Appending mask to message!")
        data["mask"] = mask

    client.publish(topic="seg_img", payload=utils.encode_cmd(data), retain=False)
    logger.debug("seg time: %f", time.time()-t1)
    if userdata["viz"]:
        cv2.circle(image, (config.XY_TRACK_POINT[0], config.XY_TRACK_POINT[1]), 10, (0, 255, 0), 1)
        cv2.imshow("mask", utils.decode_numpy_array(mask))
        cv2.imshow("image", image)
        cv2.waitKey(1)

    if latents and not args["final_latent"]:
        userdata["debugger"].add(small_ims, latents)
        userdata["debugger"].cluster()


def init_img_topic(client):
    client.on_message = _receive_img
    client.on_connect = _subscribe_img_topic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='net_seg', required=True,
                        help='Name this client/slave announces to the server. Needs to match on server side.')
    parser.add_argument('--ip', '-ip', type=str, default='', required=False,
                        help='Name this client/slave announces to the server. Needs to match on server side.')
    parser.add_argument('-viz', action='store_true', help="Activate this to enable visualization output")
    args = parser.parse_args()


    size = (480, 640)
    logger.info("Starting engine")
    engine = create_engine()
    logger.info("Starting Latent Engine")
    latent_engine = create_latent_engine()
    debg = debugger()

    # Setup Engine
    broker_data = {
        "size": size,
        "engine": engine,
        "latent_engine": latent_engine,
        "viz": args.viz,
        "debugger": debg
    }
    if args.ip:
        config.BROKER_IP = args.ip
    else:
        config.BROKER_IP = utils.find_master(args.name)

    # Defined userdata pkg we give to MMQT
    logger.info("Broker: Connecting")
    client = mqtt.Client(userdata=broker_data)
    client.connect(config.BROKER_IP, config.BROKER_PORT, keepalive=60)

    logger.info("Initialize IMG Client Functions")
    init_img_topic(client)

    # Endless loop that keeps sending all sensor properties and updates once
    # per loop the actuator
    logger.info("Entering Forever Loop")
    client.loop_forever()

    client.disconnect()

if __name__ == "__main__":
    main()
