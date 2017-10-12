#!/usr/bin/env python
"""Acts as master controlloing an EV3robot (or other types) remotely to sort lego bricks.
"""
__author__ = "Lucia Seitz"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Lucia Seitz, David Adrian"]
__license__ = "MIT"
__maintainer__ = "Lucia Seitz, David Adrian"

import time
import json
import logging
import argparse
import os
from multiprocessing import Queue, Process

import cv2
import ev3dev.ev3 as ev3
import numpy as np
from scipy import ndimage

import gdk.jetson.broker as Broker
# import gdk.jetson.sensors as Sensors
import gdk.jetson.controller as Controller
import gdk.jetson.tracker as Tracker
import gdk.jetson.marker_tracker as MarkerTracker
import gdk.config as config
import gdk.common as common
import gdk.utils as utils
import gdk.statemachine.statemachine as Statemachine

from sklearn import mixture

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center, get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper

# Currently up here, cause OnAVOSold causes conflicts with the logger
# DO NOT MOVE THIS BELOW THE SEG ENGINE IMPORT!!!
common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# from gdk.network_interface import image_to_latents

import random as rand

# from ext.OnAVOS.LatentSpace import create_latent_engine
# from ext.OnAVOS.segmenter import create_engine

seg_q = Queue(maxsize=1)


def reset_seg_q():
    if seg_q.full():
        _ = seg_q.get()


def go_to_angle(robot, angle, n_angles):
    logger.info("Going back to angle " + str(angle))
    for i in range(n_angles - angle):
        robot.controller.turn_in_place(-45, 600)
        time.sleep(1)


def read_good_image(robot, segment_engine, latent_engine, visualize=False, network_code=True):
    n_igs = 10
    n_rnd = int(n_igs / 2)
    count_good = 0
    good_latents = np.zeros((n_igs, 32))
    frame = robot.cam["ps3_one"].read()
    if visualize:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
    frames = np.zeros((n_igs, 480, 640, 3))
    logger.info("Please start juggling bricks now!")
    for i in range(n_igs):
        logger.info(str(i))
        for _ in range(15):
            frame = robot.cam["ps3_one"].read()
        frames[i] = frame
        time.sleep(0.1)

    rands = np.random.choice(n_igs, n_rnd, replace=False)
    rand_frames = frames[rands]
    logger.info("You can stop. I calculate the latents now.")
    for j in range(n_rnd):
        logger.debug("Processing latent input image: %d", j)
        if network_code:
            logger.debug("reseting queue")
            reset_seg_q()
            logger.debug("sending image to network segmenter")
            robot.controller.broker.net_segment_image(rand_frames[j], compute_latent=True)
            logger.debug("wait for new image")
            most_recent_segmentation = seg_q.get()
            logger.debug("got new image")
            latents = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
            coords = most_recent_segmentation["coords"]
            if len(latents) > 1 or len(latents) < 1:
                continue
        else:
            latents, coords, small_ims, mask, elapsed = image_to_latents(rand_frames[j], segment_engine, latent_engine,
                                                                         scale=1, occlude=False, mask_image=False)
            if len(small_ims) > 1 or len(small_ims) < 1:
                continue
        good_latents[count_good] = latents[0]
        # if visualize:
        #    cv2.imshow('frame', small_ims[0])
        # path = '/home/dlrc/lucia/acc34/frames/good/good_image'+str(count_good)+'.png'
        # cv2.imwrite(path,small_ims[0])
        count_good += 1
    good_latents = good_latents[:count_good, :]
    logger.debug(good_latents)
    return good_latents


def search_and_reach_brick_nearest(robot, segment_engine, latent_engine, config, visualize=False, record=False,
                           network_code=True):
    for _ in range(5):
        frame = robot.cam["ps3_one"].read()

    x = None
    y = None
    print("Calling image_to_latens")
    if network_code:
        reset_seg_q()
        #robot.controller.broker.net_segment_image(frame, compute_latent=True)
        #most_recent_segmentation = seg_q.get()
        #latents = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
        #coords = most_recent_segmentation["coords"]

        robot.controller.broker.net_segment_image(frame, compute_latent=False)
        most_recent_segmentation = seg_q.get()
        coords = most_recent_segmentation["coords"]
    else:
        latents, coords, small_ims, mask, elapsed = image_to_latents(frame, segment_engine, latent_engine, scale=0.5,
                                                                     occlude=True, mask_image=False)

    try:
        #TODO change to nearest brick or so
        x, y = coords[0]
        print("Best candidate:", x, y)

        if x is not None:
            errx = (x - config.XY_TRACK_POINT[0])
            erry = (y - config.XY_TRACK_POINT[1])
            logger.debug("Error: x=%f y=%f", errx, erry)

            if abs(errx) < 50 and abs(erry) < 30:
                try:
                    search_and_reach_brick_nearest.track_count += 1
                except:
                    search_and_reach_brick_nearest.track_count = 0

                if search_and_reach_brick_nearest.track_count > 5:
                    return False
                    #return False, Latents

                print("New track count:", search_and_reach_brick_nearest.track_count)

            robot.x_PID.update(erry)
            robot.w_PID.update(errx)
            v_l, v_r = robot.controller.twist_to_vel(robot.x_PID.output, robot.w_PID.output)
            print(v_l, v_r)
            robot.controller.move(v_l, v_r)
        else:
            robot.controller.move(0, 0)
    except:
        print("Exception: Probably didnt find any candidate brick.")
        robot.controller.move(0, 0)

    if visualize:
        if x is not None:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 1)
            cv2.circle(frame, (config.XY_TRACK_POINT[0], config.XY_TRACK_POINT[1]), 10, (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    #return True, latents
    return True

def is_brick_in_gripper(robot, img, engine, network_code=True):
    img_c = np.array(img[335:460, 220:350])
    size = (img_c.shape[0], img_c.shape[1])
    if network_code:
        reset_seg_q()
        robot.controller.broker.net_segment_image(img_c, compute_latent=True, return_mask=True)
        most_recent_segmentation = seg_q.get()
        mask = utils.decode_numpy_array(most_recent_segmentation["mask"])
        latent = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
    else:
        mask, _ = engine.segment(img_c, size, save_flag=False)
    if np.mean(mask) < 0.01:
        return False, None
    else:
        return True, latent


def search_box(robot, tracker, marker_ID, visualize=False):
    for _ in range(3):
        frame = robot.cam["jetson_onboard"].read()
    if visualize:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    ids = tracker.update(frame)
    logger.info("Found IDs: " + str(ids))
    if ids is not None:
        if marker_ID in tracker.update(frame):
            return False

    # Just rotate in one direction
    robot.controller.move(-60, +60)
    # time.sleep(1.5)
    # robot.controller.stop_main_motors()

    return True  # not found yet


def drive_to_box(robot, tracker, marker_ID, visualize=False, state="first_cam"):
    for _ in range(3):
        if state == "first_cam":
            frame = robot.cam["jetson_onboard"].read()
        elif state == "second_cam":
            frame = robot.cam["ps3_one"].read()

    if visualize:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    is_tracked = tracker.update(frame)
    if marker_ID in is_tracked:
        errx, erry = tracker.get_centroid_error(marker_ID, state)
        errx = utils.scale_to_range(errx, -320, 320, -100, 100)
        erry = utils.scale_to_range(erry, -240, 240, -100, 100)

        errd = tracker.get_distance_error(marker_ID)
        logger.debug("Distance error:%f", errd)

        robot.x_PID_ch.update(errd * 500)
        robot.w_PID_ch.update(errx)
        if state == "first_cam":
            if abs(errd) < 0.68:
                logger.debug("switching camera")
                return "second_cam"
        elif state == "second_cam":
            if abs(errd) < 0.77:
                logger.debug("switching camera")
                return "inside_box"

        robot.last_rotation_direction = robot.w_PID_ch.output

        logger.debug("%f, %f", robot.x_PID_ch.output,
                     robot.w_PID_ch.output)
        v_l, v_r = robot.controller.twist_to_vel(
            robot.x_PID_ch.output, robot.w_PID_ch.output)

        robot.controller.move(v_l, v_r)
        # robot.controller.move(0, 0)

        if visualize:
            if state == "first_cam":
                visualize_point_tracking(frame, (tracker.centroid[2][0], tracker.centroid[2][1]), config.XY_TRACK_POINT_MARKER, name='frame')
            elif state == "second_cam":
                visualize_point_tracking(frame, (tracker.centroid[2][0], tracker.centroid[2][1]), config.XY_TRACK_POINT_MARKER_2, name='frame')

    else:
        robot.controller.move(0, 0)
        return "lost"

    return state

def visualize_point_tracking(frame, xy, target_xy, name='frame'):
    cv2.circle(frame, xy, 10, (255, 0, 0), 1)
    cv2.circle(frame, target_xy, 10, (255, 0, 255), 1)
    cv2.imshow(name, frame)
    cv2.waitKey(1)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-viz', action='store_true', help="Activate this to enable visualization output")
    parser.add_argument('-p_rec', type=str,
                        help='Path to store recorded images. If defined, it automatically actives saving.')
    parser.add_argument('--jetson', '-j', action='store_true',
                        help="This parameter should be set if run from a Jetson board with internal camera")
    parser.add_argument('--expected_slaves', '-es', required=True, nargs='+', type=str,
                        help="The (list) of clients which need to connect to the server before processing can start.")
    parser.add_argument('--interface_robot', '-ir', required=True, type=str,
                        help="Interface to which ev3 is connect.")
    parser.add_argument('--interface_netseg', '-in', required=True, type=str,
                        help="Interface to which ev3 is connect.")
    parser.add_argument('--angle', '-a', required=False, type=int, default=8,
                        help="Interface to which ev3 is connect.")
    args = parser.parse_args()

    network_code = True

    if not utils.master_announcement(args.expected_slaves, args.interface_netseg,
                                     message_delay=3, repeats=2):
        pass
        # exit()
    if not utils.master_announcement(args.expected_slaves, args.interface_robot,
                                     message_delay=3, repeats=2):
        pass

    # else:
    config.BROKER_IP = utils.find_interface_ip(args.interface_netseg)

    # sleep shortly so every service can get ready...ye sucks
    time.sleep(2)

    # Start the MMQT Messaging system
    logger.info("MMQT Broker: Initalization")
    broker = Broker.Broker(
        config.BROKER_IP, config.BROKER_PORT, broker_data={"seg_q": seg_q})
    logger.info("Broker: Connecting")
    broker.connect()
    broker.init_img_topic()
    broker.start_listen()

    controller = Controller.RobotController(broker)
    robot = EV3Robot(controller, jetson_mode=args.jetson)

    logger.info("Initializing Robot")
    robot.startup()

    # Set up engines
    logger.info("Starting Segmentation Engine")
    segment_engine = None
    logger.info("Starting Latent Engine")
    latent_engine = None
    logger.info("Setting up tracker Engine")
    marker_tracker = MarkerTracker.MarkerTracker()

    logger.info("Starting Control Loop")


    logger.info("Starting Brick Scanning Phase")

    robot.cam["ps3_one"].open()
    robot.cam["jetson_onboard"].open()
    robot.controller.lift_gripper()

    # #### Get images
    logger.info("Starting Environment Exploration Phase")
    path = '/home/nvidia/Documents/Ach8/Raw_ims/'
    count = 1
    markerIDs = set()
    for round_num in range(16):
        robot.controller.turn_in_place(-28, 1000)

        start = time.time()
        elapsed = time.time() - start
        while elapsed < 0.5:
            for i in range(7):
                frame = robot.cam["ps3_one"].read()
                frame2 = robot.cam["jetson_onboard"].read()
            ids = marker_tracker.update(frame2)
            logger.debug("ids: " + str(ids))
            for id in ids:
                markerIDs.add(id[0])
            name = str(count).zfill(4)
            cv2.imwrite(path + name + '.png', frame)
            count += 1
            elapsed = time.time() - start
    robot.cam["jetson_onboard"].close()

    ##### Set up engines
    if not network_code:
        logger.info("Starting Segmentation Engine")
        # segment_engine = create_small_engine()
        segment_engine = create_engine()
        logger.info("Starting Latent Engine")
        # latent_engine = create_small_latent_engine()
        latent_engine = create_latent_engine()

    ##### Get latents
    num_latent_var = 32
    small_im_path = '/home/nvidia/Documents/Ach8/Small_ims/'
    mask_path = '/home/nvidia/Documents/Ach8/Masks/'
    all_latents = np.empty([0, num_latent_var])
    for j, im_filename in enumerate(sorted(os.listdir(path))):
        logger.debug("Reading image from "+ path + im_filename)
        frame = cv2.imread(path + im_filename, cv2.IMREAD_COLOR)

        if network_code:
            reset_seg_q()
            robot.controller.broker.net_segment_image(frame, compute_latent=True)
            most_recent_segmentation = seg_q.get()
            latents = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
            logger.debug(str(latents))
            coords = most_recent_segmentation["coords"]
        else:
            latents, coords, small_ims, mask, elapsed = image_to_latents(frame, segment_engine, latent_engine, scale=1, occlude=False, mask_image=False)

        for i, latent in enumerate(latents):
            all_latents = np.concatenate((all_latents, latents[i]), axis=0)
        #print("image " + str(j) + " / " + str(len(os.listdir(path))) + " elapsed: " + str(elapsed) + " num objects: " + str(len(latents)))
    logger.debug(str(len(latents))+ " bricks found.")
    latents = np.array(all_latents)
    print(latents.shape)
    np.save('/home/nvidia/Documents/Ach8/latents', latents)

    #### Known k clustering
    tot_analysis_start = time.time()
    # latents = np.load('/home/dlrc/Documents/Ach7/latents.npy')

    n_components = len(markerIDs)
    logger.debug("Found %d boxes.", n_components)
    gmm = mixture.GaussianMixture(n_components=n_components,covariance_type='full')
    gmm.fit(latents)
    # cluster_labels_mix = gmm.predict(latents)

    logger.info("All_analysis finished, elapsed: " + str(time.time() - tot_analysis_start))

    ##Go to nearest brick

    picking_bricks = True
    robot.controller.lower_gripper()
    while picking_bricks:
        running = True
        latents = []
        while running:
            #running, latent = search_and_reach_brick_nearest(robot, segment_engine, latent_engine, config, visualize=args.viz,record=False)
            #latents.append(latents)
            running = search_and_reach_brick_nearest(robot, segment_engine, latent_engine, config, visualize=args.viz,record=False)

	

        cluster_labels_mix = gmm.predict(np.array(latents))
        label = int(np.mean(cluster_labels_mix))


        robot.controller.stop_main_motors()

        #logger.info("Starting PickUp")
        #robot.controller.move(60, 60)
        #time.sleep(0.2)
        #robot.controller.stop_main_motors()
        robot.controller.close_gripper()
        robot.controller.lift_gripper()

        for _ in range(15):
            frame = robot.cam["ps3_one"].read()

        picking_bricks,latent = not is_brick_in_gripper(robot, frame, segment_engine)
        # picking_bricks = False
        if picking_bricks:
            logger.info("Oh no! I lost the brick!")
            robot.controller.move(-45, -45)
            time.sleep(0.5)

            robot.controller.open_gripper()
            robot.controller.lower_gripper()
            # time.sleep(2)

        
    cluster_labels_mix = gmm.predict(np.array(latent))
    label = int(np.mean(cluster_labels_mix))    

    robot.cam["ps3_one"].close()

    # Brick is picked up
    # Robot moves towards box
    robot.controller.lift_gripper()
    robot.cam["jetson_onboard"].open()
    markerIDs_list = list(markerIDs)
    markerID = markerIDs_list[label]
    running = True
    logger.info("Start looking for box with ID " + str(markerID) + " now!")
    while running:
        running = search_box(robot, marker_tracker, markerID)
        logger.debug("Running.")

    robot.controller.stop_main_motors()
    logger.info("Found box.")
    robot.cam["jetson_onboard"].close()

    # Now actually track it!
    running = True
    robot.cam["jetson_onboard"].open()
    logger.info("Driving to box " + str(markerID) + " now!")
    state="first_cam"
    switched=False

    while running:
        last_state = state
        state = drive_to_box(robot, marker_tracker, markerID, args.viz, state)
        if state == "second_cam" and not switched:
            logger.debug("Box-Search: Switching to ps3 cam")
            robot.cam["jetson_onboard"].close()
            robot.cam["ps3_one"].open()
            robot.controller.move(65,65)
            time.sleep(0.5)
            robot.controller.stop_main_motors()
            switched = True
        elif state == "lost":
            logger.debug("lost marker")
            running = True
            state = last_state
        elif state == "inside_box":
            logger.debug("inside box")
            running = False
        #     searching = True
        #     while searching:
        #         searching = search_home_base(robot, tracker, config, args.viz)

    if state == "inside_box":
        robot.controller.stop_main_motors()
        robot.controller.open_gripper()
        robot.controller.stop_main_motors()
        robot.cam["jetson_onboard"].close()

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
