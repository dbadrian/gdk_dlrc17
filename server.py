#!/usr/bin/env python
"""Acts as master controlloing an EV3robot (or other types) remotely to sort lego bricks.
"""
__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import time
import json
import logging
import argparse
import os

import cv2
import ev3dev.ev3 as ev3
import numpy as np
from scipy import ndimage

import gdk.jetson.broker as Broker
# import gdk.jetson.sensors as Sensors
import gdk.jetson.controller as Controller
import gdk.jetson.tracker as Tracker
import gdk.config as config
import gdk.common as common
import gdk.utils as utils
import gdk.statemachine.statemachine as Statemachine

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center,get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper

# Currently up here, cause OnAVOSold causes conflicts with the logger
common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ext.OnAVOS.segmenter import create_engine

def make_square_box(sx, sy):
    # calculate sizes of the box
    xx = sx.stop - sx.start
    yy = sy.stop - sy.start
    # the function asumes xx>yy, so the code below reverses variables if the case is opposite
    if yy > xx:
        (xx, yy, sy, sx) = (yy, xx, sx, sy)
        ref = 480
        big = 'y'
    else:
        ref = 640
        big = 'x'
    # check the padding
    if (xx - yy) % 2 == 0:
        new_y = (sy.start - (xx - yy) / 2, sy.stop + (xx - yy) / 2)
    if (xx - yy) % 2 == 1:
        new_y = y_slice = (sy.start - (xx - yy - 1) / 2, sy.stop + (xx - yy - 1) / 2 + 1)
    # check the borders
    if new_y[0] <= 0:
        new_y = (0, new_y[1] + abs(new_y[0]))
    elif new_y[1] >= ref:
        new_y = (new_y[0] - (new_y[1] - ref), ref)
    # return new slices
    if big == 'x':
        return sx, slice(int(new_y[0]), int(new_y[1]), None)
    if big == 'y':
        return slice(int(new_y[0]), int(new_y[1]), None), sx

def get_bounding_box(mask):
    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # Now that we have only one connect component, extract it's bounding box
    try:
        slice_x, slice_y, _ = ndimage.find_objects(label_im == 1)[0]
    except:
        print ("err" + str(len(label_im==1)))
        # slice_x, slice_y, _ = ndimage.find_objects(label_im == 1)[0]
        return None,None
    # roi = mask[slice_x, slice_y]

    new_slice_x, new_slice_y = make_square_box(slice_x, slice_y)
    # roi2 = mask[new_slice_x, new_slice_y]

    return new_slice_x, new_slice_y

def segment_image(engine, frame, occlude_gripper=False, size=(240, 320)):
    if occlude_gripper:
        frame = oclude_gripper(frame)
    return engine.segment(frame, size, save_flag=True)


def momentum_scaling(old, new, p=0.85):
    if not old:
        old = new
    new_scaled = int(p*new + (1-p)*old)
    return new_scaled


def detect_brick(engine, frame):
    start = time.time()
    size = (240, 320)
    mask, prob = segment_image(engine, frame, size=size)
    logger.debug("Segmentation %d", time.time() - start)
    x, y = center(mask, size)

    if x and y:
        is_tracked = True
        x = utils.scale_to_range(x, 0, size[1], 0, frame.shape[1])
        y = utils.scale_to_range(y, 0, size[0], 0, frame.shape[0])
    else:
        is_tracked = False

    return is_tracked, x, y, mask


def visualize_point_tracking(frame, xy, target_xy, name='frame'):
    cv2.circle(frame, xy, 10, (255, 0, 0), 1)
    cv2.circle(frame, target_xy, 10, (255, 0, 255), 1)
    cv2.imshow(name, frame)
    cv2.waitKey(1)


def control_loop(desired_control_frequency):
    def control_loop_decorator(func):
        def func_wrapper(*args, **kwargs):
            control_loop_start = time.time()

            res = func(*args, **kwargs)

            diff = time.time() - control_loop_start
            if diff > (1.0/desired_control_frequency):
                logger.debug("Missed desired control frequecy (%f) = %f", desired_control_frequency, 1.0/diff)
            else:
                sleep_time = np.abs((1.0/desired_control_frequency) - diff)
                logger.debug("Faster than desired control frequency - going to sleep: %f", sleep_time)
                time.sleep(sleep_time)

            return res
        return func_wrapper
    return control_loop_decorator


@control_loop(config.DESIRED_CONTROL_HZ)
def search_and_reach_brick(robot, engine, config, visualize=False, record=False):

    # def reset():
    #     search_and_reach_brick.track_count = 0
    #     search_and_reach_brick.prev_x = None
    #     search_and_reach_brick.prev_y = None

    try:
        if(search_and_reach_brick.rec_count + 1 > 200):
            return False,False
    except:
        search_and_reach_brick.rec_count = 0

    for _ in range(15):
        frame = robot.cam["ps3_one"].read()

    is_tracked, x, y, mask = detect_brick(engine, frame)
    # is_tracked = False

    # Only if we got a tracked point, we continue. Else: Stop any
    # movement
    if is_tracked:
        # Improve tracking by considering the momentum of the tracked point

        try:
            search_and_reach_brick.nothing_in_view = 0
        except AttributeError:
            search_and_reach_brick.nothing_in_view = 0

        try:
            x = momentum_scaling(search_and_reach_brick.prev_x, x)
            search_and_reach_brick.prev_x = x
        except AttributeError:
            search_and_reach_brick.prev_x = x

        try:
            y = momentum_scaling(search_and_reach_brick.prev_y, y)
            search_and_reach_brick.prev_y = y
        except AttributeError:
            search_and_reach_brick.prev_x = x

        logger.debug("x=%d, y=%d", x, y)

        # Calculate error between current location and desired target
        # location
        # /(config.XY_TRACK_POINT[0])
        errx = (x - config.XY_TRACK_POINT[0])
        # /(config.XY_TRACK_POINT[1])
        erry = (y - config.XY_TRACK_POINT[1])
        logger.debug("%d, %d", errx, erry)
        # Reached Center Zone, drop out of this behavior part

        try:
            if search_and_reach_brick.track_count > 5:
                return True,True
        except AttributeError:
            search_and_reach_brick.track_count = 0

        if abs(errx) < 50 and abs(erry) < 30:
            search_and_reach_brick.track_count = search_and_reach_brick.track_count + 1
            logger.debug("New track count: %d", search_and_reach_brick.track_count)


        errx = utils.scale_to_range(errx, -config.XY_TRACK_POINT[1]/2.0,
                                    config.XY_TRACK_POINT[1]/2.0, -100, 100)
        erry = utils.scale_to_range(erry, -config.XY_TRACK_POINT[0]/2.0,
                                    config.XY_TRACK_POINT[0]/2.0, -100, 100)

        # Update the PID controller
        robot.x_PID.update(erry)
        robot.w_PID.update(errx)

        # Generate Motor Signals from the differential drive controller
        logger.debug("Control Outputs: vl=%d vr=%d",
                     robot.x_PID.output, robot.w_PID.output)
        v_l, v_r = robot.controller.twist_to_vel(
            robot.x_PID.output, robot.w_PID.output)

        # Send signal to the robot/slave
        robot.controller.move(v_l, v_r)

    else:
        robot.controller.move(0, 0)
        try:
            search_and_reach_brick.nothing_in_view += 1
        except:
            search_and_reach_brick.nothing_in_view = 0

        if search_and_reach_brick.nothing_in_view > 5:
            robot.controller.move(90,-80)

        # try:
        #     search_and_reach_brick.nothing_in_view += 1
        #     logger.debug("Nothing seen %d", search_and_reach_brick.nothing_in_view)
        #     if search_and_reach_brick.nothing_in_view>5:
        #         # robot.cam["ps3_one"].close()
        #         robot.cam["ps3_two"].open()
        #         frame = robot.cam["ps3_two"].read()
        #         is_tracked, x, y, mask = detect_brick(engine, frame)
        #         if is_tracked:

        #             errx = (x - config.XY_TRACK_POINT[0])
        #             erry = (y - config.XY_TRACK_POINT[1])
        #             logger.debug("%d, %d", errx, erry)

        #             errx = utils.scale_to_range(errx, -config.XY_TRACK_POINT[1] / 2.0,
        #                                         config.XY_TRACK_POINT[1] / 2.0, -100, 100)
        #             erry = utils.scale_to_range(erry, -config.XY_TRACK_POINT[0] / 2.0,
        #                                         config.XY_TRACK_POINT[0] / 2.0, -100, 100)

        #             robot.x_PID.update(erry)
        #             robot.w_PID.update(errx)
        #             # Generate Motor Signals from the differential drive controller
        #             v_l, v_r = robot.controller.twist_to_vel(
        #                 robot.x_PID.output, robot.w_PID.output)

        #             logger.debug("Control Outputs: vl=%d vr=%d",
        #                          robot.x_PID.output, robot.w_PID.output)

        #             # Send signal to the robot/slave
        #             robot.controller.move(v_l, v_r)
        #     else:
        #         # Halt robot
        #         robot.controller.move(0, 0)
        #         robot.cam["ps3_two"].close()
        #     # robot.cam["ps3_one"].open()
        #     if search_and_reach_brick.nothing_in_view > 25:
        #         robot.controller.move(-45, 45)
        # except AttributeError:
        #     search_and_reach_brick.nothing_in_view = 0

    # Visualize the tracking and target
    if visualize:
        if is_tracked:
            # cv2.circle(mask, (xold, yold), 10, (0, 0, 255), 1)
            # visualize_point_tracking(frame, (int(x), int(y)), config.XY_TRACK_POINT, "tracked_point")
            cv2.imshow('mask', mask)
        else:
            cv2.imshow('mask', mask)
            cv2.imshow('tracked_point', frame)
            cv2.waitKey(1)

    if record:
        size = (240, 320)
        image = cv2.resize(frame, (320,240))
        x, y = center(mask, size)
        contours = get_objects_seg(mask)

        if not x:
            print('no brick')
            # cv2.imshow('frame', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            return True,True

        ret, seperate_masks = cv2.connectedComponents(mask)
        print(seperate_masks.max().max())

        orig_mask = mask
        for ii in range(1, len(contours) + 1):
            bool_mask = (seperate_masks == ii)
            mask = orig_mask * bool_mask

            mask = np.stack((mask,) * 3, 2)
            bool_mask = mask > mask.mean()

            masked_im = image * bool_mask

            slice_x, slice_y = get_bounding_box(mask)
            if slice_x is None:
                continue
            small_im = masked_im[slice_x, slice_y]
            small_im = cv2.resize(small_im, (64, 64))

            # cv2.imshow('frame', image)
            # cv2.imshow('mask' + str(ii), small_im)
            # cv2.imshow('hm', small_im)

            # latent = get_latent(small_im, sess, loc)

            # print(latent)
            # try:
            search_and_reach_brick.rec_count += 1
            # except AttributeError:
            #     search_and_reach_brick.rec_count = 0

            name = str(1).zfill(5) + '_' + str(search_and_reach_brick.rec_count).zfill(4)
            cv2.imwrite(os.path.join(record, name + '.png'), small_im)



        # try:
        #     search_and_reach_brick.rec_count += 1
        # # cv2.imwrite('/home/dlrc/mask2.jpg', mask)
        # except AttributeError:
        #     search_and_reach_brick.rec_count = 0
        #
        # cv2.imwrite(os.path.join(record, 'frame' + str(search_and_reach_brick.rec_count) + '.jpg'), frame)

    # Brick not reached yet, continue loop
    return True, True


@control_loop(30)
def search_home_base(robot, tracker, config, visualize=True):
    for _ in range(3):
        frame = robot.cam["ps3_two"].read()
    if tracker.update(frame):
        return False

    if visualize:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    # Just rotate in one direction
    robot.controller.move(-45, +45)
    # time.sleep(1.5)
    # robot.controller.stop_main_motors()

    return True  # not found yet


@control_loop(config.DESIRED_CONTROL_HZ)
def drive_to_home_base(robot, tracker, config, visualize=True):
    for _ in range(15):
        frame = robot.cam["ps3_two"].read()

    if visualize:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    is_tracked = tracker.update(frame)
    if is_tracked:
        errx, erry = tracker.get_centroid_error()
        errx = utils.scale_to_range(errx, -320, 320, -100, 100)
        erry = utils.scale_to_range(erry, -240, 240, -100, 100)

        errd = tracker.get_distance_error()
        logger.debug("Distance error:%f", errd)

        robot.x_PID_ch.update(errd*500)
        robot.w_PID_ch.update(errx)
        if abs(errd) < 0.76:
            logger.debug("Inside box")
            return "Inside"
        robot.last_rotation_direction = robot.w_PID_ch.output

        logger.debug("%f, %f", robot.x_PID_ch.output,
                     robot.w_PID_ch.output)
        v_l, v_r = robot.controller.twist_to_vel(
            robot.x_PID_ch.output, robot.w_PID_ch.output)

        robot.controller.move(v_l, v_r)

        if visualize:
            visualize_point_tracking(frame, (tracker.centroid[0][0], tracker.centroid[0][1]), config.XY_TRACK_POINT, name='frame')
    else:
        robot.controller.move(0, 0)
        return "Lost"

    return "Driving"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-viz', action='store_true', help="Activate this to enable visualization output")
    parser.add_argument('-p_rec', type=str, help='Path to store recorded images. If defined, it automatically actives saving.')
    parser.add_argument('--jetson', '-j', action='store_true',
                        help="This parameter should be set if run from a Jetson board with internal camera")
    parser.add_argument('--expected_slaves', '-es', required=True, nargs='+', type=str,
                        help="The (list) of clients which need to connect to the server before processing can start.")
    parser.add_argument('--interface','-i', required=True, type=str,
                        help="Interface to which ev3 is connect.")
    args = parser.parse_args()

    if not utils.master_announcement(args.expected_slaves, args.interface,
                                     message_delay=5, repeats=2):
        pass
        # exit()
    # else:
    config.BROKER_IP = utils.find_interface_ip(args.interface)
    time.sleep(2)


    # Start the MMQT Messaging system
    logger.info("MMQT Broker: Initalization")
    broker = Broker.Broker(
        config.BROKER_IP, config.BROKER_PORT, broker_data={})
    logger.info("Broker: Connecting")
    broker.connect()

    controller = Controller.RobotController(broker)
    robot = EV3Robot(controller, jetson_mode=args.jetson)
    statemachine = Statemachine.RobotBrain("Ben")

    logger.info("Initializing Robot")
    robot.startup()

    # Setup Segmenter
    logger.info("Starting Segmentation Engine")
    engine = create_engine()
    logger.info("Setting uo tracker Engine")
    tracker = Tracker.CheckerBoardTracker()

    logger.info("Starting Control Loop")
    # while True:
        # reset what needs to be reset
        # robot.controller.lower_gripper()
    robot.controller.lift_gripper()
    try:
        delattr(search_and_reach_brick, "track_count")
        delattr(search_and_reach_brick, "prev_x")
        delattr(search_and_reach_brick, "prev_y")
    except:
        pass

    logger.info("Starting Search Phase")
    running = True
    robot.cam["ps3_one"].open()
    while running:
        running,found = search_and_reach_brick(robot, engine, config, visualize=args.viz, record=args.p_rec)

    robot.controller.stop_main_motors()
    robot.cam["ps3_one"].close()

    logger.info("Starting PickUp")
    robot.controller.lower_gripper()
    robot.controller.close_gripper()
    robot.controller.lift_gripper()

    robot.controller.open_gripper()
    time.sleep(2)

    # Check if Checkerboard "HOME" is in view, otherwise rotate till found
    robot.cam["ps3_two"].open()
    logger.info("Looking for checkerboard")
    running = True
    while running:
        running = search_home_base(robot, tracker, config, args.viz)

    robot.controller.stop_main_motors()
    robot.cam["ps3_two"].close()

    # Now actually track it!
    running = True
    robot.cam["ps3_two"].open()
    while running:
        state = drive_to_home_base(robot, tracker, config, args.viz)
        if state == "Inside":
            running = False
        elif state == "Lost":
            pass
        #     searching = True
        #     while searching:
        #         searching = search_home_base(robot, tracker, config, args.viz)

    robot.controller.stop_main_motors()
    robot.controller.open_gripper()
    robot.controller.stop_main_motors()
    robot.cam["ps3_two"].close()

    logger.info("Looking for new work opportunities")
    robot.controller.move(-60, -60)
    time.sleep(2)
    robot.controller.stop_main_motors()
    robot.controller.move(60, -60)
    time.sleep(4)
    robot.controller.stop_main_motors()

    print("shutting down...im so sleepy")

if __name__ == "__main__":
    main()
