__author__ = "David Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian, Richard Kurle"]
__license__ = "MIT"
__maintainer__ = "David Adrian"

import time

import cv2

import gdk.jetson.broker as broker
import gdk.jetson.controller as Controller
import gdk.jetson.sensors as Sensors


class EV3Robot:
    def __init__(self, controller, jetson_mode=False):
        self.jetson_mode = jetson_mode

        # # PID for Pick Up
        # #self.x_PID = Controller.PID(20, 10, 13.5)
        # self.x_PID = Controller.PID(15, 8, 10.5)
        # self.x_PID.setWindup(200)
        # self.w_PID = Controller.PID(0.15, 0.12, 0.12    )
        # self.w_PID.setWindup(15)

        self.x_PID = Controller.PID(14, 16.5, 16)
        self.x_PID.setWindup(220)
        self.w_PID = Controller.PID(0.18, 0.21, 0.21)
        self.w_PID.setWindup(8)


        # PID for checkerboard tracking
        self.x_PID_ch = Controller.PID(9, 0.2, 0)
        self.x_PID.setWindup(100)
        self.w_PID_ch = Controller.PID(1.0, 0.15, 0.1)
        self.last_rotation_direction = None

        # Controller Sending Motor Signals to EV3
        self.controller = controller
        self.is_gripper_open = None
        self.init = False

        # Initialize Cameras
        if self.jetson_mode:
            cam_jetson_local = Sensors.OnBoardCamera()
        else:
            cam_jetson_local = None

        # In case of Jetson mode, we need to shift index by one
        self.cam = {
            "jetson_onboard": cam_jetson_local,
            "ps3_one": Sensors.ExternalCamera(0 + int(self.jetson_mode)),
            "ps3_two": Sensors.ExternalCamera(1 + int(self.jetson_mode))
        }

        # self.cam_ps3_one = Sensors.ExternalCamera(0 + int(self.jetson_mode))
        # self.cam_ps3_two = Sensors.ExternalCamera(1 + int(self.jetson_mode))


    def startup(self):
        # Bring Gripper into a well-defined position
        self.controller.reset()
        self.controller.stop_main_motors()
        self.controller.lift_gripper()
        self.controller.close_gripper()
        self.controller.open_gripper()
        self.controller.lower_gripper()
        # time.sleep(2)
        self.is_gripper_open = True

        # Init-Phase finished
        self.init = True


    # State Functions
    def __is_gripper_open(self):
        return self.is_gripper_open
