__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import logging
import json

import paho.mqtt.client as mqtt
import ev3dev.ev3 as ev3

from gdk.jetson.sensors import Ev3Sensor
import gdk.config as config
import gdk.common as common
import gdk.utils as utils

logger = logging.getLogger(__name__)

class Broker(object):
    """
    This class listens on the configured port to messages from ev3 and sets the
    corresponding entries in the sensor class.
    The Broker also sends out messages to the ev3 actuators.
    """
    def __init__(self, ip_address, port, broker_data):
        self.client = mqtt.Client(userdata=broker_data)
        self.client.on_connect = self._subscribe_to_topics
        self.client.on_message = self.__process_cmd
        self.ip_address = ip_address
        self.port = port

        logger.debug("Initalization IP=%s PORT=%s", self.ip_address, self.port)

    # Define function that is called on connecting: Subscribes to all sensor topics
    def _subscribe_to_topics(self, client, userdata, flags, rc):  # on_connect
        pass

    def __process_cmd(self, client, userdata, msg):  # on_message
        pass

    def _receive_img(self, client, userdata, msg):
        logger.debug("Got segmented image")
        data = json.loads(msg.payload.decode())
        userdata["seg_q"].put((data))


    def _subscribe_img_topic(self, client, userdata, flags, rc):
        logger.debug("Connected with result code "+str(rc))
        logger.debug("Subscribing to topic=seg_img")
        self.client.subscribe("seg_img")


    def init_img_topic(self):
        self.client.on_message = self._receive_img
        self.client.on_connect = self._subscribe_img_topic


    def start_listen(self):
        print("start listening...")
        self.client.loop_start()

    def stop_listen(self):
        print("stop listening...")
        self.client.loop_stop()

    def send_message(self, cmd):
        logger.debug("Publish (topic=cmd): %s", cmd)
        self.client.publish(topic="cmd", payload=utils.encode_cmd(cmd))

    def poll_sensor(self, name):
        logger.debug("Publish (topic=%s): read_sensor", name)
        self.client.publish(topic=name, payload=utils.encode_cmd(("read_sensor",{"name": name})))

    def connect(self):
        logger.debug("Connecting IP=%s PORT=%s KEEP_ALIVE=60", self.ip_address, self.port)
        self.client.connect(self.ip_address, self.port, keepalive=60)

    def disconnect(self):
        logger.debug("Disconnecting")
        self.client.disconnect()

    def net_segment_image(self, img, compute_latent=False, return_mask=False, scale=1.0, occlude=False, final_latent=False):
        cmd_raw = ("net_segment_image", {"image": utils.encode_numpy_array(img),
                                         "compute_latent": compute_latent,
                                         "return_mask": return_mask,
                                         "scale": scale,
                                         "occlude": occlude,
                                         "final_latent": final_latent})
        logger.debug("Publish (topic=img): %s", ["img", {"compute_latent": compute_latent,
                                         "return_mask": return_mask,
                                         "scale": scale}])
        self.client.publish(topic="img", payload=utils.encode_cmd(cmd_raw))
