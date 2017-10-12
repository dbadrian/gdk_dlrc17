__author__ = "David Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian, Richard Kurle"]
__license__ = "MIT"
__maintainer__ = "David Adrian"

BROKER_IP = None # WILL BE SET AUTOMATICALLY!
BROKER_PORT = 1883
SERVICE_BROADCAST_PORT = 51244  # Port to announce master on
SERVICE_ACK_PORT = 51245  # Port to ack master on
SERVICE_BROADCAST_MAGIC = "gdk51244"  # Magic value to verify correct master

# Define Connected Motors and where they are connected (or should be connected)
actuators_h = [
    {"type": "LARGE_MOTOR",
     "port": "outA",
     "name": "Left_Motor"},
    {"type": "LARGE_MOTOR",
     "port": "outB",
     "name": "Right_Motor"},
    {"type": "LARGE_MOTOR",
     "port": "outC",
     "name": "Gripper_Motor"},
    {"type": "MEDIUM_MOTOR",
     "port": "outD",
     "name": "Gripper_Locking_Motor"},
]

sensors_h = [
    {"type": "TOUCH_SENSOR",
     "port": "in2",
     "name": "Gripper_Lift_Sensor"},
    {"type": "TOUCH_SENSOR",
     "port": "in1",
     "name": "Gripper_Open_Sensor"},
]

DIST_BETWEEN_WHEELS = 200  # in MM
WHEEL_CIRCUMFERENCE = 147  # in MM

DESIRED_CONTROL_HZ = 20

SEG_SIZE = (480, 854)

# CheckerBoard Configurations
PATTERN_SIZE = (4, 5)
BEST_DISTANCE = [460, 800]

# Segmenation Tracker
PS3_CAM_RESOLUTION = (480, 854)  # pair, Y-X res
#XY_TRACK_POINT = (369, 450)  # ps3 cam
XY_TRACK_POINT = (339, 400)  # ps3 cam
XY_TRACK_POINT_MARKER = (640, 550)  # jetson
XY_TRACK_POINT_MARKER_2 = (320, 260)
# XY_TRACK_POINT = (290, 480)  # ps3 cam
