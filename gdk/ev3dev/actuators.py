__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import logging
import json

import paho.mqtt.client as mqtt

import gdk.ev3dev.controller as controller
import gdk.common as common

logger = logging.getLogger(__name__)

def _subscribe_cmd_topic(client, userdata, flags, rc):
    logger.debug("Connected with result code "+str(rc))
    logger.debug("Subscribing to topic=cmd")
    client.subscribe("cmd")


def _process_command(client, userdata, msg):
    func, args = json.loads(msg.payload.decode())
    logger.debug("Calling {}, with args={}".format(func, args))
    result = getattr(controller, func)(**args)


def initialize_actuator_client_functions(client):
    client.on_message = _process_command
    client.on_connect = _subscribe_cmd_topic
