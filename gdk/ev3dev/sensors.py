__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import paho.mqtt.client as mqtt
import gdk.config as config
import ev3dev.ev3 as ev3


import logging
import time
import math

import ev3dev.ev3 as ev3

import gdk.config as config
import gdk.utils as utils
import gdk.common as common

logger = logging.getLogger(__name__)


class Sensors(object):
    """
    This class translate higher-level motor commands to low-level instructions.
    """

    def __init__(self, sensor_list):
        self.sensors = {}
        # self.sensors_and_names_dict = {} # actuator_name: actuator for actuator_name, actuator in zip(self.actuator_names, self.sensors)
        self.main_motors_running = False

        for sensor in sensor_list:
            if sensor["type"] == "TOUCH_SENSOR":
                logger.debug("Adding TOUCH_SENSOR:%s", sensor["port"])
                self.sensors[sensor["name"]] = ev3.TouchSensor(sensor["port"])
            elif sensor["type"] == "INFRARED_SENSOR":
                logger.debug("Adding MEDIUM_MOTOR:%s", sensor["port"])
                self.sensors[sensor["name"]] = ev3.InfraredSensor(sensor["port"])
            elif sensor["type"] == "COLOR_SENSOR":
                logger.debug("Adding MEDIUM_MOTOR:%s", sensor["port"])
                self.sensors[sensor["name"]] = ev3.ColorSensor(sensor["port"])


    def read_sensor(self, name):
        sensor = self.sensors[name]
        msgs = []
        for property_name, dtype in sensor.__class__.__dict__.items():
            if isinstance(dtype, property):
                property_value = getattr(sensor, property_name)
                msgs.append("{}".format(property_value))
        return msgs