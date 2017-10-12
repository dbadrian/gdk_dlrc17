__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import logging
import time
import math

import ev3dev.ev3 as ev3

import gdk.config as config
import gdk.utils as utils
import gdk.common as common

logger = logging.getLogger(__name__)


class Controller(object):
    """
    This class translate higher-level motor commands to low-level instructions.
    """

    def __init__(self, actuator_list, sensors):
        self.actuators = {}
        # self.actuators_and_names_dict = {} # actuator_name: actuator for actuator_name, actuator in zip(self.actuator_names, self.actuators)
        self.main_motors_running = None
        self.gripper_closed = None
        self.gripper_up = None
        self.sensors = sensors
        self.gripper_up = int(self.sensors.read_sensor("Gripper_Lift_Sensor")[0])

        for actuator in actuator_list:
            if actuator["type"] == "LARGE_MOTOR":
                logger.debug("Adding LARGER_MOTOR:%s", actuator["port"])
                self.actuators[actuator["name"]] = ev3.LargeMotor(actuator["port"])
            elif actuator["type"] == "MEDIUM_MOTOR":
                logger.debug("Adding MEDIUM_MOTOR:%s", actuator["port"])
                self.actuators[actuator["name"]] = ev3.MediumMotor(actuator["port"])

            # Additional Generic Configurations
            self.actuators[actuator["name"]].reset()


    def reset(self):
        self.main_motors_running = None
        self.gripper_closed = None
        self.gripper_up = None


    def move_distance(self, distance, speed):
        # Move in a straight line for specified distance in cm (only used for final move towards target)

        self.actuators["Left_Motor"].reset()
        self.actuators["Right_Motor"].reset()

        wheel_angle = (distance / config.WHEEL_CIRCUMFERENCE) * 180.0

        setattr(self.actuators["Left_Motor"], 'position_sp', int(wheel_angle))
        setattr(self.actuators["Right_Motor"], 'position_sp', int(wheel_angle))

        setattr(self.actuators["Left_Motor"], 'speed_sp', speed)
        setattr(self.actuators["Right_Motor"], 'speed_sp', speed)

        setattr(self.actuators["Left_Motor"], 'command', 'run-to-rel-pos')
        setattr(self.actuators["Right_Motor"], 'command', 'run-to-rel-pos')


    def turn_in_place(self, turn_angle, speed):
        # Allows rotation in place
        # pos angle: counter-clockwise turn (mathematical sense)
        # Turn in place (velocity_right = -velocity_left) by specified degrees
        # $ echo 180 > $MC/position_sp
        # $ echo run-to-rel-pos > $MC/command
        self.actuators["Left_Motor"].reset()
        self.actuators["Right_Motor"].reset()

        wheel_distance = (turn_angle / 180.0) * math.pi * config.DIST_BETWEEN_WHEELS
        wheel_angle = (wheel_distance / config.WHEEL_CIRCUMFERENCE) * 180.0
        logger.debug("Wheel Angle Turn (in deg): %s", wheel_angle)

        setattr(self.actuators["Left_Motor"], 'position_sp', -int(wheel_angle))
        setattr(self.actuators["Right_Motor"], 'position_sp', int(wheel_angle))

        setattr(self.actuators["Left_Motor"], 'speed_sp', speed)
        setattr(self.actuators["Right_Motor"], 'speed_sp', speed)

        setattr(self.actuators["Left_Motor"], 'command', 'run-to-rel-pos')
        setattr(self.actuators["Right_Motor"], 'command', 'run-to-rel-pos')


        # logger.debug("Starting Sleep delay of t=%d", abs(wheel_angle/speed))
        # time.sleep(abs(wheel_angle/speed))
        # logger.debug("Stopping Motors")
        # self.stop(["Left_Motor", "Right_Motor"])


    def move(self, speed_left, speed_right):
        speed_left = utils.clamp(speed_left, -85, 85)
        speed_right = utils.clamp(speed_right, -85, 85)
#        ls = 25
#        if speed_left > 5 and speed_left < ls:
#            speed_left = ls
#        if speed_left < -5 and speed_left >-ls:
#            speed_left = -ls
#        if speed_right > 5 and speed_right < ls:
#            speed_right = ls
#        if speed_right < -5 and speed_right >-ls:
#            speed_right = -ls

        logger.debug("Changing Speed left=%d, right=%d", speed_left, speed_right)
        setattr(self.actuators["Left_Motor"], 'duty_cycle_sp', int(speed_left))
        setattr(self.actuators["Right_Motor"], 'duty_cycle_sp', int(speed_right))

        if not self.main_motors_running:
            setattr(self.actuators["Left_Motor"], 'command', 'run-direct')
            setattr(self.actuators["Right_Motor"], 'command', 'run-direct')
            self.main_motors_running = True


    def stop_main_motors(self):
        logger.debug("Running: command=stop on main motors")
        setattr(self.actuators["Left_Motor"], 'command', 'stop')
        setattr(self.actuators["Right_Motor"], 'command', 'stop')
        self.main_motors_running = False


    def lift_gripper(self):
        # status = int(self.sensors.read_sensor("Gripper_Lift_Sensor")[0])
        status = False
        logger.debug("Gripper lift status=%s", status)
        if not self.gripper_up:# not status or
            # logger.debug("Gripper is lowered, lifting now!")

            # setattr(self.actuators["Gripper_Motor"], 'duty_cycle_sp', 35)
            # setattr(self.actuators["Gripper_Motor"], 'command', 'run-direct')

            # count = 0
            # while not int(self.sensors.read_sensor("Gripper_Lift_Sensor")[0]):
            #     time.sleep(0.05)
            #     count += 1
            #     speed = utils.clamp(35 + count, -99, 99)
            #     setattr(self.actuators["Gripper_Motor"], 'duty_cycle_sp', speed)
            # # time.sleep(0.1) # tighten it further

            setattr(self.actuators["Gripper_Motor"], 'stop_action', 'hold')
            setattr(self.actuators["Gripper_Motor"], 'command', 'stop')
            setattr(self.actuators["Gripper_Motor"], 'position_sp', +45)
            setattr(self.actuators["Gripper_Motor"], 'speed_sp', 60)
            setattr(self.actuators["Gripper_Motor"], 'command', 'run-to-rel-pos')

            self.gripper_up = True

        else:
            logger.debug("Gripper is lifted, doing nothing mate!")


    def lower_gripper(self):
        status = int(self.sensors.read_sensor("Gripper_Lift_Sensor")[0])
        logger.debug("Gripper lift status=%s", status)
        if status or self.gripper_up:
            logger.debug("Gripper is lifted, lowering now!")
            setattr(self.actuators["Gripper_Motor"], 'stop_action', 'hold')
            setattr(self.actuators["Gripper_Motor"], 'position_sp', -33)
            setattr(self.actuators["Gripper_Motor"], 'speed_sp', 100)
            setattr(self.actuators["Gripper_Motor"], 'command', 'run-to-rel-pos')
            # setattr(self.actuators["Gripper_Motor"], 'duty_cycle_sp', 50)
            # setattr(self.actuators["Gripper_Motor"], 'command', 'run-direct')

            # time.sleep(0.1)

            # setattr(self.actuators["Gripper_Motor"], 'command', 'stop')

            self.gripper_up = False

        else:
            logger.debug("Gripper is lowered, doing nothing mate!")


    def open_gripper(self):
        status = int(self.sensors.read_sensor("Gripper_Open_Sensor")[0])
        logger.debug("Gripper claw status=%s", status)
        if not status or self.gripper_closed:
            logger.debug("Gripper is closed, opening now!")
            setattr(self.actuators["Gripper_Locking_Motor"], 'duty_cycle_sp', 90)
            setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'run-direct')

            while not int(self.sensors.read_sensor("Gripper_Open_Sensor")[0]):
                time.sleep(0.1)
            time.sleep(0.1) # tighten it further

            setattr(self.actuators["Gripper_Locking_Motor"], 'stop_action', 'hold')
            setattr(self.actuators["Gripper_Locking_Motor"], 'position_sp', +10)
            setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'stop')
            setattr(self.actuators["Gripper_Locking_Motor"], 'speed_sp', 900)
            setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'run-to-rel-pos')

            self.gripper_closed = False

        else:
            logger.debug("Gripper is open, doing nothing mate!")


    def close_gripper(self):
        status = int(self.sensors.read_sensor("Gripper_Open_Sensor")[0])
        logger.debug("Gripper claw status=%s", status)
        if status or not self.gripper_closed:
            logger.debug("Gripper is open, closing now!")

            setattr(self.actuators["Gripper_Locking_Motor"], 'stop_action', 'hold')
            setattr(self.actuators["Gripper_Locking_Motor"], 'position_sp', -1300)
            setattr(self.actuators["Gripper_Locking_Motor"], 'speed_sp', 900)
            setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'run-to-rel-pos')

            # setattr(self.actuators["Gripper_Locking_Motor"], 'duty_cycle_sp', 50)
            # setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'run-direct')
            # setattr(self.actuators["Gripper_Locking_Motor"], 'command', 'stop')

            self.gripper_closed = True


        else:
            logger.debug("Gripper is closed, doing nothing mate!")


    def stop(self, actuators):
        # Stop motors
        for actuator in actuators:
            logger.debug("Running %s:command=stop",
                         self.actuators[actuator])
            setattr(self.actuators[actuator], 'command', 'stop')
