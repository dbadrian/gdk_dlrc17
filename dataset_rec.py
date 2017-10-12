__author__ = "David B. Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David B. Adrian"

import os
import argparse

import cv2
import numpy as np

from gdk.robot import EV3Robot
from gdk.imagproc.centroid_finder import center
from gdk.imagproc.preprocessing import oclude_gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', required=True, type=int,
                    help="cam id")
    parser.add_argument('--out', '-o', required=True, type=str,
                    help="path to save dataset")
    args = parser.parse_args()

    assert os.path.isdir(args.out), "path doesnt exist"

    cam = cv2.VideoCapture(args.id)

    recording = False

    count_dataset = 0
    img_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    kp = ord("a")
    while True:
        ret, frame = cam.read()
        frame_viz = np.copy(frame)

        name = str(img_count).zfill(4)

        if kp == ord("r"):
            if not recording:
                path = os.path.join(args.out, str(count_dataset))
                recording = True
                os.makedirs(path)
                count_dataset +=1
                cv2.imwrite(os.path.join(path, name)+ '.png', frame)

        elif kp == ord("s") and recording:
            recording = False
            img_count = 0

        elif kp == ord("c"):
            exit()

        if recording:
            img_count += 1
            cv2.imwrite(os.path.join(path, name)+ '.png', frame)
            cv2.putText(frame_viz,'REC',(10,500), font, 4,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow("Camera output:",frame_viz)
        kp = cv2.waitKey(5) & 0xFF

if __name__ == "__main__":
    main()