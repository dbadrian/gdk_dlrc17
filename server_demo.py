#!/usr/bin/env python
"""Acts as master controlling an EV3robot (or other types) remotely to sort lego bricks.
"""
__author__ = "Lucia Seitz"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Lucia Seitz, David Adrian, Jonathon Luiten, Karoline Stosio"]
__license__ = "MIT"
__maintainer__ = "Lucia Seitz"

import time
import json
import logging
import argparse
import os
from multiprocessing import Queue, Process


import cv2
import ev3dev.ev3 as ev3
import numpy as np
import scipy.stats as nps
from scipy import ndimage
from sklearn import mixture


import gdk.jetson.broker as Broker
# import gdk.jetson.sensors as Sensors
import gdk.jetson.controller as Controller
import gdk.jetson.tracker as Tracker
import gdk.jetson.marker_tracker as MarkerTracker
import gdk.config as config
import gdk.common as common
import gdk.utils as utils
import gdk.statemachine.statemachine as Statemachine
from debugger import debugger

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center,get_objects_seg
from gdk.imagproc.preprocessing import oclude_gripper
import gdk.imagproc.multitracking as multitracking

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


def go_to_angle(robot, angle, n_angles):
    logger.info("Going back to angle " + str(angle))
    for i in range(n_angles - angle):
        robot.controller.turn_in_place(-45,600)
        time.sleep(1)


def read_good_image(robot,segment_engine,latent_engine,visualize=False,network_code=True):
    n_igs = 50
    n_rnd = int(n_igs/2)
    count_good = 0
    good_latents = np.zeros((n_igs, 32))
    frame = robot.cam["ps3_one"].read()
    if visualize:
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
    frames = np.zeros((n_igs, 480,640,3))
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
            latents,coords,small_ims, mask, elapsed = image_to_latents(rand_frames[j],segment_engine,latent_engine,scale=1,occlude=False,mask_image=False)
            if len(small_ims) > 1 or len(small_ims) < 1:
                continue
        good_latents[count_good] = latents[0]
        #if visualize:
        #    cv2.imshow('frame', small_ims[0])
        # path = '/home/dlrc/lucia/acc34/frames/good/good_image'+str(count_good)+'.png'
        # cv2.imwrite(path,small_ims[0])
        count_good += 1
    good_latents = good_latents[:count_good,:]
    logger.debug(good_latents)
    return good_latents


@control_loop(config.DESIRED_CONTROL_HZ)
def search_and_reach_brick(robot, segment_engine, latent_engine, config, active_trackers, new_boxes, tracked_brick_small_ims, frames_since_last_seg, visualize=False, record=False, network_code=True):
    # for _ in range(5):
    frame = robot.cam["ps3_one"].read()
        # frames_since_last_seg.append(frame)

    # Used to store the x/y coords of tracked brick
    x = None
    y = None

    # Get new segmentations from the network segmenter and reinit trackers
    if network_code:
        if seg_q.full():
            t = time.time()
            logger.debug("Got new segmentation image back!")
            # Get most recent segmented image
            most_recent_segmentation = seg_q.get()
            # Publish most recent segmented image to process
            robot.controller.broker.net_segment_image(frame, compute_latent=False, return_mask=False, scale=0.5, occlude=True)
            logger.debug("Sending new image took %f sec", time.time() - t)
            # REINIT TRACKER HERE with recent image seg output
            frames_since_last_seg.append(frame)  #append current frame so it will be used for update as well!

            X,Y=config.XY_TRACK_POINT
            centroids = [multitracking.get_center(bbox) for bbox in most_recent_segmentation["boxes"]]
            distances = [np.sqrt((c[0]-X)**2 + (c[1]-Y)**2) for c in centroids]
            if distances:
                idx = np.argmin(distances)

                del active_trackers[:]
                # active_trackers.append(cv2.TrackerMedianFlow_create())
                active_trackers.append(cv2.TrackerKCF_create())

                ok = active_trackers[0].init(frame, tuple(most_recent_segmentation["boxes"][idx]))
                if not ok:
                    logger.debug("Could not reinit tracker")
                    # robot.controller.move(0, 0)
                    return "lost"
                t = time.time()
                # Playback of frames
                for frm in frames_since_last_seg:
                    active_trackers, new_boxes = multitracking.update_active_trackers(active_trackers, frm)

                if new_boxes:
                    tracked_brick_small_ims.append(multitracking.get_img_from_bbox(frame, new_boxes[0]))

                del frames_since_last_seg[:]
                logger.debug("Playback of frames took %f sec", time.time() - t)
        else:
            logger.debug("Updating tracker without segmentation!")
            active_trackers, new_boxes = multitracking.update_active_trackers(active_trackers, frame)

        # append all frame for playback once new frame arrives // after a reset the first one will be the same as the returned mask
        frames_since_last_seg.append(frame)
    else:
        print("Calling image_to_latens")
        latents,coords,small_ims,mask, elapsed = image_to_latents(frame,segment_engine,latent_engine,scale=0.5,occlude=True,mask_image=False)
        #TODO MISSING LOCAL CODE!
        raise NotImplemented

    # Check if we lost our tracking!
    if len(active_trackers) != 1:
        logger.debug("got %d active_trackers", len(active_trackers))
        robot.controller.move(0, 0)
        return "lost"

    centroid = [multitracking.get_center(bbox) for bbox in new_boxes][0]
    x = centroid[0]
    y = centroid[1]

    if visualize:
        cv2.circle(frame, centroid, 10, (255, 0, 0), 1)

    print("Best candidate:", x, y)
    try:

        if x is not None:
            errx = (x - config.XY_TRACK_POINT[0])
            erry = (y - config.XY_TRACK_POINT[1])
            logger.debug("Error: x=%f y=%f", errx, erry)

            if abs(errx) < 40 and abs(erry) < 10:
                try:
                    search_and_reach_brick.track_count += 1
                except:
                    search_and_reach_brick.track_count = 0

                if search_and_reach_brick.track_count > 5:
                        return "reached"

                print("New track count:", search_and_reach_brick.track_count)


            robot.x_PID.update(erry)
            robot.w_PID.update(errx)
            v_l, v_r = robot.controller.twist_to_vel(robot.x_PID.output, robot.w_PID.output)
            logger.debug("%f %f", v_l, v_r)
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

    return "running"

def is_brick_in_gripper(robot, img, engine, network_code=True):
    img_c = np.array(img[335:460,220:350])
    size = (img_c.shape[0],img_c.shape[1])
    if network_code:
        reset_seg_q()
        robot.controller.broker.net_segment_image(img_c, compute_latent=False, return_mask=True)
        most_recent_segmentation = seg_q.get()
        mask = utils.decode_numpy_array(most_recent_segmentation["mask"])
    else:
        mask, _ = engine.segment(img_c, size, save_flag=False)
    if np.mean(mask)  < 0.01: return False
    else: return True

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
    robot.controller.move(-65, +65)
    # time.sleep(1.5)
    # robot.controller.stop_main_motors()

    return True  # not found yet

def drive_to_box(robot, tracker, marker_ID, visualize=False, state="first_cam"):
    # for _ in range(3):
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
        logger.debug("X error:%f", errx)

        if abs(errd) < 0.66:
            errd = 0.01
        robot.x_PID_ch.update(errd*500)
        robot.w_PID_ch.update(errx)
        if state == "first_cam":
            if abs(errd) < 0.69 and abs(errx) < 25:
                # logger.debug("switching camera")
                # return "second_cam"
                logger.debug("inside box")
                return "inside_box"

        # elif state == "second_cam":
        #     if abs(errd) < 0.77:
        #         logger.debug("switching camera")
        #         return "inside_box"

        robot.last_rotation_direction = robot.w_PID_ch.output

        logger.debug("%f, %f", robot.x_PID_ch.output,
                     robot.w_PID_ch.output)
        v_l, v_r = robot.controller.twist_to_vel(
            robot.x_PID_ch.output, robot.w_PID_ch.output)

        robot.controller.move(v_l, v_r)
        #robot.controller.move(0, 0)

        if visualize:
            if state == "first_cam":
                visualize_point_tracking(frame, (tracker.centroid[marker_ID][0], tracker.centroid[marker_ID][1]), config.XY_TRACK_POINT_MARKER, name='frame')
            elif state == "second_cam":
                visualize_point_tracking(frame, (tracker.centroid[marker_ID][0], tracker.centroid[marker_ID][1]), config.XY_TRACK_POINT_MARKER_2, name='frame')

    else:
        robot.controller.move(0, 0)
        if state == "first_cam":
            return "first_cam"
        else:
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
    parser.add_argument('-b', action='store_true', help="Activate this to enable broadcasting")
    parser.add_argument('-p_rec', type=str, help='Path to store recorded images. If defined, it automatically actives saving.')
    parser.add_argument('--jetson', '-j', action='store_true',
                        help="This parameter should be set if run from a Jetson board with internal camera")
    parser.add_argument('--expected_slaves', '-es', required=True, nargs='+', type=str,
                        help="The (list) of clients which need to connect to the server before processing can start.")
    parser.add_argument('--interface_robot','-ir', required=True, type=str,
                        help="Interface to which ev3 is connect.")
    parser.add_argument('--interface_netseg','-in', required=True, type=str,
                        help="Interface to which ev3 is connect.")
    parser.add_argument('--angle','-a', required=False, type=int, default=16,
                        help="Interface to which ev3 is connect.")
    args = parser.parse_args()

    network_code = True

    if args.b:
        if not utils.master_announcement(args.expected_slaves, args.interface_netseg,
                                          message_delay=3, repeats=2):
            pass
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
    #segment_engine = create_engine()
    segment_engine = None
    logger.info("Starting Latent Engine")
    #latent_engine = create_latent_engine()
    latent_engine = None
    logger.info("Setting up tracker Engine")
    marker_tracker = MarkerTracker.MarkerTracker()



    #######################################################
    # ALL THE OTHER CODE FOR EXPLORATION AND SHIT IN HERE #
    #######################################################
    robot.cam["ps3_one"].open()
    robot.cam["jetson_onboard"].open()
    # #### Get images
    logger.info("Starting Environment Exploration Phase")
    # path = '/home/nvidia/Documents/Ach8/Raw_ims/'

    count = 1
    markerIDs = set()
    brick_exp_frames = []
    for round_num in range(args.angle):
        robot.controller.turn_in_place(+28, 1000)

        # start = time.time()
        # elapsed = time.time() - start

        for i in range(6):
            frame = robot.cam["ps3_one"].read()
            frame2 = robot.cam["jetson_onboard"].read()
            if i % 2 == 0:
                brick_exp_frames.append(frame)
                # brick_exp_frames.append(frame2)

        ids = marker_tracker.update(frame2)
        logger.debug("ids: " + str(ids))
        for id in ids:
            markerIDs.add(id[0])
        # name = str(count).zfill(4)
        # cv2.imwrite(path + name + '.png', frame)
        # count += 1

        # while elapsed < 0.5:
        #     for i in range(7):
        #         frame = robot.cam["ps3_one"].read()
        #         frame2 = robot.cam["jetson_onboard"].read()
        #     ids = marker_tracker.update(frame2)
        #     logger.debug("ids: " + str(ids))
        #     for id in ids:
        #         markerIDs.add(id[0])
        #     name = str(count).zfill(4)
        #     cv2.imwrite(path + name + '.png', frame)
        #     count += 1
        #     elapsed = time.time() - start


    robot.cam["jetson_onboard"].close()
    robot.cam["ps3_one"].close()

    ##### Set up engines
    if not network_code:
        logger.info("Starting Segmentation Engine")
        # segment_engine = create_small_engine()
        segment_engine = create_engine()
        logger.info("Starting Latent Engine")
        # latent_engine = create_small_latent_engine()
        latent_engine = create_latent_engine()
    else:
        segment_engine = None
        latent_engine = None

    ##### Get latents
    num_latent_var = 32
    all_latents = np.empty([1, num_latent_var])
    # for j, im_filename in enumerate(sorted(os.listdir(path))):
    for j, frame in enumerate(brick_exp_frames):
        # logger.debug("Reading image from "+ path + im_filename)
        logger.debug("Computing latents for frame %d", j)
        # frame = cv2.imread(path + im_filename, cv2.IMREAD_COLOR)

        if network_code:
            reset_seg_q()
            robot.controller.broker.net_segment_image(frame, compute_latent=True,  occlude=True)
            most_recent_segmentation = seg_q.get()
            latents = utils.decode_list_of_numpy_arrays(most_recent_segmentation["latents"])
            logger.debug(str(latents))
            coords = most_recent_segmentation["coords"]
        else:
            latents, coords, small_ims, mask, elapsed = image_to_latents(frame, segment_engine, latent_engine, scale=1, occlude=False, mask_image=False)

        for i, latent in enumerate(latents):
            all_latents = np.concatenate((all_latents, latents[i]), axis=0)
        #print("image " + str(j) + " / " + str(len(os.listdir(path))) + " elapsed: " + str(elapsed) + " num objects: " + str(len(latents)))
    latents = np.array(all_latents)
    logger.debug(str(latents.shape[0]) + " scans of bricks.")
    logger.debug(str(latents.shape) + "latent dim.")
    #np.save('/home/nvidia/Documents/Ach8/latents', latents)

    if latents.shape[0] < 3*len(markerIDs):
        logger.error("WARNING NOT ENOUGH BRICK SCANS FOUND FOR GOOD ESTIMATION")

    n_components = len(markerIDs)
    logger.debug("Found %d boxes.", n_components)
    dbg = debugger()
    gmm = mixture.GaussianMixture(n_components=n_components,covariance_type='full')
    gmm.fit(latents)

    # logger.info("All_analysis finished, elapsed: " + str(time.time() - tot_analysis_start))




    #######################################################
    # ACTUAL BRICK SORTING COMES NOW #
    #######################################################
    logger.info("Starting Brick Sorting Loop")
    while True:
        try:
            logger.debug("reseting Search-and-Reach Brick loop function")
            delattr(search_and_reach_brick, "track_count")
            delattr(search_and_reach_brick, "prev_x")
            delattr(search_and_reach_brick, "prev_y")
        except:
           pass


        logger.info("Starting Search-and-Reach Brick Phase")
        robot.cam["ps3_one"].open()
        picking_bricks = True
        robot.controller.lower_gripper()

        # Get the net segmenter ready for processing
        #
        # This list will keep images of the tracked brick which we will use later to compute the mean latent feature
        tracked_brick_small_ims = []

        # This list can be used to store all frames since the last tracking update through segmentation for playback
        frames_since_last_seg = []

        # Read first frame
        frame = robot.cam["ps3_one"].read()
        # send it to the net_seg
        logger.debug("SRB: Send initial image to network segmenter")
        reset_seg_q()
        robot.controller.broker.net_segment_image(frame, compute_latent=False, return_mask=True, scale=0.5, occlude=True)
        #  wait for results
        most_recent_segmentation = seg_q.get()
        logger.debug(most_recent_segmentation.keys())
        # and start tracker for closest brick
        logger.debug("SRB: Instantiate tracker for closest brick.")
        if most_recent_segmentation["coords"]:
            active_trackers, new_boxes = multitracking.instantiate_trackers(utils.decode_numpy_array(most_recent_segmentation["mask"]), frame, track_closest=True)
            tracked_brick_small_ims.append(multitracking.get_img_from_bbox(frame, new_boxes[0]))
            # Update tracker and send off next frame to keep the loop running! IMPORTANT!
        else:
            active_trackers = []
            new_boxes = []

        frame = robot.cam["ps3_one"].read()
        logger.debug("SRB: Send second, loop starting image to network segmenter")
        robot.controller.broker.net_segment_image(frame, compute_latent=False, return_mask=True, scale=0.5, occlude=True)
        # active_trackers, new_boxes = multitracking.update_active_trackers(active_trackers, frame)
        # tracked_brick_small_ims.append(multitracking.get_img_from_bbox(frame, new_boxes[0]))


        # robot.controller.lift_gripper()
        while picking_bricks:
            running = True
            while running:
                logger.debug("SRB: starting main routine")
                tim = time.time()
                state = search_and_reach_brick(robot, segment_engine, latent_engine, config, active_trackers, new_boxes, tracked_brick_small_ims, frames_since_last_seg, visualize=args.viz, record=False)
                logger.info("SBR took: %f", time.time()-tim)
                if state == "lost":
                    logger.debug("Lost track of brick, restarting search")
                elif state == "reached":
                    running = False

            picking_bricks = False


            robot.controller.stop_main_motors()


            tbsi = len(tracked_brick_small_ims)
            logger.debug("SRB: reached brick, computing latents from %d mini images", len(tracked_brick_small_ims))
            brick_latents = []
            for idx, im in enumerate(tracked_brick_small_ims):  # TODO: RANDOM SAMPLE?
                logger.debug("SRB: reached brick, processing frame %d", idx)
                robot.controller.broker.net_segment_image(np.array(im), compute_latent=True, return_mask=False, final_latent=True, occlude=True)
                most_recent_segmentation = seg_q.get()
                if most_recent_segmentation["latents"]:
                    brick_latents.append(utils.decode_numpy_array(most_recent_segmentation["latents"][0]))

            for idx, im in enumerate(tracked_brick_small_ims[-10:-1]):
                robot.controller.broker.net_segment_image(np.array(tracked_brick_small_ims[idx]), compute_latent=True, return_mask=False, final_latent=True, occlude=True)
                most_recent_segmentation = seg_q.get()
                if most_recent_segmentation["latents"]:
                    brick_latents.append(utils.decode_numpy_array(most_recent_segmentation["latents"][0]))


            reset_seg_q()
            frame = robot.cam["ps3_one"].read()
            robot.controller.broker.net_segment_image(frame, compute_latent=True, return_mask=True, scale=0.5, occlude=True)
            #  wait for results
            most_recent_segmentation = seg_q.get()
            logger.debug(most_recent_segmentation.keys())
            # and start tracker for closest brick
            logger.debug("SRB: Instantiate tracker for closest brick.")
            if most_recent_segmentation["coords"]:
                active_trackers, new_boxes = multitracking.instantiate_trackers(utils.decode_numpy_array(most_recent_segmentation["mask"]), frame, track_closest=True)
                tracked_brick_small_ims.append(multitracking.get_img_from_bbox(frame, new_boxes[0]))
                robot.controller.broker.net_segment_image(tracked_brick_small_ims[0], compute_latent=True, return_mask=True, scale=0.5, occlude=True)
                brick_latents.append(utils.decode_numpy_array(most_recent_segmentation["latents"][0]))

            # for _ in range(5):
            #     frame = robot.cam["ps3_one"].read()
            #     robot.controller.broker.net_segment_image(frame, compute_latent=True, return_mask=False)
            #     most_recent_segmentation = seg_q.get()
            #     if most_recent_segmentation["latents"]:
            #         brick_latents.append(utils.decode_numpy_array(most_recent_segmentation["latents"][0]))


            # delete the stuff
            del tracked_brick_small_ims[:]


# ############################################################
            # COMPUTE LATENT MEAN HERE
            # DECIDE WHICH CLUSTER!
############################################################
            logger.debug(np.array(brick_latents).shape)
            bricks_latents_arry = np.array(brick_latents)
            print(np.array(brick_latents))
            cluster_labels_mix = gmm.predict(bricks_latents_arry[:,0,:])
            logger.debug(bricks_latents_arry[:,0,:].shape)
            logger.debug(cluster_labels_mix.shape)
            label,count = nps.mode(cluster_labels_mix, axis=0)
            label = label[0]
            logger.debug("desired label %s:", label)
            logger.debug(cluster_labels_mix)
            # label = int(np.mode(cluster_labels_mix))

            logger.info("Starting PickUp")
            # robot.controller.move(60, 60)
            # time.sleep(0.2)

            for _ in range(30):
                frame = robot.cam["ps3_one"].read()

            #picking_bricks = not is_brick_in_gripper(robot, frame, segment_engine)
            picking_bricks = False
            if picking_bricks:
                logger.info("Oh no! I lost the brick!")
                robot.controller.move(-45, -45)
                time.sleep(0.5)

                robot.controller.open_gripper()
                robot.controller.lower_gripper()


        robot.cam["ps3_one"].close()

        robot.controller.lower_gripper()
        robot.controller.close_gripper()
        robot.controller.lift_gripper()

        # Brick is picked up
        # Robot moves towards box
        markerIDs_list = list(markerIDs)
        markerID = markerIDs_list[label]


        robot.cam["jetson_onboard"].open()
        running = True
        logger.info("Start looking for box with ID " + str(markerID) + " now!")
        while running:
            running = search_box(robot, marker_tracker, markerID)
            logger.debug("Running.")

        robot.controller.stop_main_motors()
        logger.info("Found box.")
        # robot.cam["jetson_onboard"].close()

        # Now actually track it!
        running = True
        # robot.cam["jetson_onboard"].open()
        logger.info("Driving to box " + str(markerID) + " now!")
        state="first_cam"
        switched=False

        while running:
            last_state = state
            state = drive_to_box(robot, marker_tracker, markerID, args.viz, state)
            logger.debug("state=%s, des_box=%d, all_box=%s, small_ims_used=%d", state, markerID, markerIDs_list, tbsi)
            if state == "second_cam" and not switched:
                logger.debug("Box-Search: Switching to ps3 cam")
                robot.cam["jetson_onboard"].close()
                robot.cam["ps3_one"].open()
                robot.controller.move(65,65)
                time.sleep(0.5)
                robot.controller.stop_main_motors()
                switched = True
            elif state == "lost":
                logger.debug("lost marker. backing up")
                robot.controller.move(-65,-65)
                time.sleep(0.5)
                robot.controller.stop_main_motors()
                time.sleep(0.5)
                running = True
                robot.cam["jetson_onboard"].open()
                robot.cam["ps3_one"].close()
                state = "first_cam"
                switched = False
            elif state == "inside_box":
                logger.debug("inside box")
                robot.cam["jetson_onboard"].close()
                robot.controller.move(60,60)
                time.sleep(0.7)
                robot.controller.stop_main_motors()
                running = False
            #     searching = True
            #     while searching:
            #         searching = search_home_base(robot, tracker, config, args.viz)

        if state == "inside_box":
            robot.controller.stop_main_motors()
            robot.controller.open_gripper()
            robot.controller.stop_main_motors()
            robot.cam["jetson_onboard"].close()

        robot.controller.open_gripper()
        logger.info("Looking for new work opportunities")
        robot.controller.move(-65, -65)
        time.sleep(4)
        robot.controller.stop_main_motors()
        robot.controller.move(80, -80)
        time.sleep(8)
        robot.controller.stop_main_motors()


    print("shutting down...im so sleepy")

if __name__ == "__main__":
    main()
