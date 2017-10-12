__author__ = "David Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian, Richard Kurle"]
__license__ = "MIT"
__maintainer__ = "David Adrian"

import paho.mqtt.client as mqtt
import gdk.config as config
#import ev3dev.ev3 as ev3
import ctypes
import numpy as np
import sys
import cv2

class Sensor(object):
    def __init__(self, *args, **kwargs):
        pass

    def read(self):
        raise ValueError('This function must be implemented by ')


class Ev3Sensor(Sensor):
    """
    Sensor class for ev3dev sensors.
    Reads all entries for @property of the given ev3 sensor (see properties on ev3dev python git repo).
    The properties are sensor values. The broker receives these messages from ev3dev
    and updates the corresponding equally named variables of this class.
    For communication to work, ev3dev must run its main.py and a connection must be set up.
    Check out ev3dev homepage for setting up network connection and configure the correct port
    of BROKER_IP in config.py on both ev3 and jetson.

    Note: In this implementation, you cannot set sensor modes, etc., the sensor merely
    streams its properties.
    """

    def __init__(self, ev3_sensor, *args, **kwargs):
        super(Ev3Sensor, self).__init__(*args, **kwargs)
        self.ev3_sensor = ev3_sensor
        self.name = self.ev3_sensor.__str__()
        self.properties_list = []
        for member, dtype in self.ev3_sensor.__class__.__dict__.items():
            if isinstance(dtype, property):
                # add property and initialize value with None
                self.properties_list.append(member)
                setattr(self, member, None)

    def read(self):
        """
        :return: dictionary: keys are string of property name, values are the corresponding value
        """
        sensor_values = [getattr(self, prop) for prop in self.properties_list]
        return dict(zip(self.properties_list, sensor_values))


class IMU(Sensor):
    def __init__(self, path_to_shared_lib_mpu='./IOInterface/jetson/mpu/libmpu.so', bus_filename='/dev/i2c-1', bus_adresses=[0x68, 0x69]):
        bus_filename = bus_filename.encode('ascii')
        self.libmpu = ctypes.cdll.LoadLibrary(path_to_shared_lib_mpu)

        self.file_descriptors = [self.libmpu.initIMU(bus_filename, bus_adress) for bus_adress in bus_adresses]
        self.data_c_arrays = [(ctypes.c_int16*7)() for _ in range(len(bus_adresses))]
        self.name = 'imu'
        self.data_sources = ["temperature", "acceleration", "gyro"]

    def read(self):
        data_dict = {}
        for idx, (file_descriptor, data_c_array) in enumerate(zip(self.file_descriptors, self.data_c_arrays)):
            self.libmpu.readIMU(file_descriptor, data_c_array)
            data_np_array = np.array(data_c_array)
            data_dict['temperature_{}'.format(idx)] = data_np_array[0] / 340.0 + 36.53
            data_dict['acceleration_{}'.format(idx)] = np.array([int(data_np_array[1]),
                                                                 int(data_np_array[2]),
                                                                 int(data_np_array[3]),
                                                                 ])
            data_dict['gyro_{}'.format(idx)] = np.array([int(data_np_array[4]),
                                                         int(data_np_array[5]),
                                                         int(data_np_array[6]),
                                                         ])
        return data_dict

    def read_sensor_nr(self, sensor_nr):
        # TODO: Ask Max, if the magic values for temperature conversion are correct.
        data_dict = {}
        self.libmpu.readIMU(self.file_descriptors[sensor_nr], self.data_c_arrays[sensor_nr])
        data_np_array = np.array(self.data_c_arrays[sensor_nr])
        data_dict['temperature'] = data_np_array[0] / 340.0 + 36.53
        data_dict['acceleration'] = np.array([int(data_np_array[1]), int(data_np_array[2]), int(data_np_array[3])])
        data_dict['gyro'] = np.array([int(data_np_array[4]), int(data_np_array[5]), int(data_np_array[6])])
        return data_dict

    def get_data_sources(self):
        return self.data_sources


class OnBoardCamera(Sensor):
    def __init__(self):
        self.name = 'onBoardCamera'
        self.cap = cv2.VideoCapture()

    def read(self):
        if self.cap.isOpened():
            ret_val, frame = self.cap.read();
        else:
            raise ValueError('Camera not opened.')
        return frame

    def open(self):
        if not self.cap.isOpened():
            self.cap.open("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    
    def close(self):
        if self.cap.isOpened():
            self.cap.release()


class ExternalCamera(Sensor):
    def __init__(self, camera_id):
        self.name = 'ExternalCamera_' + str(camera_id)
        self.cam_id = camera_id
        self.cap = cv2.VideoCapture()

    def read(self):
        if self.cap.isOpened():
            ret_val, frame = self.cap.read();
        else:
            raise ValueError('Camera not opened.')
        return frame

    def set(self, propId, value):
        return self.cap.set(propId, value)

    def get(self, propId):
        return self.cap.get(propId)

    def open(self):
        if not self.cap.isOpened():
            self.cap.open(self.cam_id)

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
