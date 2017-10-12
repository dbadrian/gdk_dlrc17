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
import ev3dev.ev3 as ev3

import gdk.config as config
import gdk.utils as utils
from gdk.ev3dev.controller import Controller
from gdk.ev3dev.sensors import Sensors
import gdk.common as common


common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _subscribe_cmd_topic(client, userdata, flags, rc):
    logger.debug("Connected with result code "+str(rc))
    logger.debug("Subscribing to topic=cmd")
    client.subscribe("cmd")


def _process_cmd(client, userdata, msg):
    func, args = json.loads(msg.payload.decode())
    logger.debug("Calling {}, with args={}".format(func, args))
    result = getattr(userdata["controller"], func)(**args)


def initialize_actuator_client_functions(client):
    client.on_message = _process_cmd
    client.on_connect = _subscribe_cmd_topic


def _subscribe_sensor_topic(client, userdata, flags, rc):
    logger.debug("Connected with result code "+str(rc))
    for sensor in config.sensors_h:
        logger.debug("Subscribing to topic=%s", sensor["name"])
        client.subscribe(sensor["name"])


def _process_sensor(client, userdata, msg):
    func, args = json.loads(msg.payload.decode())
    logger.debug("Calling {}, with args={}".format(func, args))
    result = getattr(userdata["sensors"], func)(**args)
    # for msg in result:
    #     client.publish(topic=args["name"], payload=msg)


def initialize_sensor_client_functions(client):
    client.on_message = _process_sensor
    client.on_connect = _subscribe_sensor_topic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='motor_slave', required=True,
                        help='Name this client/slave announces to the server. Needs to match on server side.')
    args = parser.parse_args()

    config.BROKER_IP = utils.find_master(args.name)

    # Setup Hardware/Actuators
    sensors = Sensors(config.sensors_h)
    controller = Controller(config.actuators_h, sensors)

    # Defined userdata pkg we give to MMQT
    broker_data = {
        "controller": controller,
        "sensors": sensors
    }
    logger.debug("Setting Up Broker Userdata=%s", broker_data)

    logger.info("Broker: Connecting")
    client = mqtt.Client(userdata=broker_data)
    client.connect(config.BROKER_IP, config.BROKER_PORT, keepalive=60)

    logger.info("Initialize Actuator Client Functions")
    initialize_actuator_client_functions(client)
    # initialize_sensor_client_functions(client)

    # Endless loop that keeps sending all sensor properties and updates once
    # per loop the actuator
    logger.info("Entering Forever Loop")
    client.loop_forever()
    running = True
    counter = 0
    # while running:
        # client.loop(5)
        # utils.announce_slave(args.name, config.BROKER_IP)
 

    import atexit
    atexit.register(client.disconnect)


if __name__ == "__main__":
    main()
