__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import logging

import gdk.utils
import gdk.common as common
import gdk.config as config


logger = logging.getLogger(__name__)

import time


class RobotController(object):
    """
    This class wraps commands and enabled communication to the low-level controller on the robot.
    """
    def __init__(self, broker):
        self.broker = broker

    def reset(self, blocking=True):
        cmd_raw = ("reset", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(1)

    def move_fwd(self, speed, run_time):
        cmd_raw = ("move_forward_for_t", {"actuators":["Left_Motor", "Right_Motor"], "speed":speed, "run_time":run_time})
        self.broker.send_message(cmd_raw)

    def move_distance(self, distance, speed):
        cmd_raw = ("move_distance", {"distance":distance, "speed":speed})
        self.broker.send_message(cmd_raw)

    def turn_in_place(self, turn_angle, speed, blocking=True):
        cmd_raw = ("turn_in_place", {"turn_angle":turn_angle, "speed":speed})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(2)

    def move(self, speed_left, speed_right,):
        cmd_raw = ("move", {"speed_left":speed_left, "speed_right":speed_right})
        self.broker.send_message(cmd_raw)

    def stop_main_motors(self, blocking=True):
        cmd_raw = ("stop_main_motors", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(0.5)

    def lift_gripper(self, blocking=True):
        cmd_raw = ("lift_gripper", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(3)

    def lower_gripper(self, blocking=True):
        cmd_raw = ("lower_gripper", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(3)

    def release_gripper(self, blocking=True):
        cmd_raw = ("release_gripper", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(1)

    def open_gripper(self, blocking=True):
        cmd_raw = ("open_gripper", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(3.5)

    def close_gripper(self, blocking=True):
        cmd_raw = ("close_gripper", {})
        self.broker.send_message(cmd_raw)

        if blocking:
            time.sleep(5)

    def stop(self, ):
        cmd_raw = ("stop", {})
        self.broker.send_message(cmd_raw)

    def twist_to_vel(self, x, w):
        # w=(angle), x=(fwd)
        v_left = (x - w * config.DIST_BETWEEN_WHEELS / 2.0) / (config.WHEEL_CIRCUMFERENCE / 2.0)
        v_right = (x + w * config.DIST_BETWEEN_WHEELS / 2.0) / (config.WHEEL_CIRCUMFERENCE / 2.0)
        return v_left, v_right

    def vel_to_twist(self, v_left, v_right):
        vx = ((v_right + v_left) * (config.WHEEL_CIRCUMFERENCE / 2.0)) / 2;
        vth = ((v_left - v_right) / config.DIST_BETWEEN_WHEELS) * (config.WHEEL_CIRCUMFERENCE / 2.0);
        return vx, vth

class PID:

    def __init__(self, Kp=0.2, Ki=0.0, Kd=0.0):
        pass

        # Removed for now
