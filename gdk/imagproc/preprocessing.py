__author__ = "Akshat Tandon"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Akshat Tandon"]
__license__ = "MIT"

import cv2
import numpy as np
def oclude_gripper(mask):
    # gripper1_p1=(0,360)
    # gripper1_p2 = (180, 480)
    # gripper2_p1 = (420, 360)
    # gripper2_p2 = (640, 480)
    # # print("ocluding gripper")
    # cv2.rectangle(mask, gripper1_p1, gripper1_p2, (255,255,255), thickness=-1 , lineType=8, shift=0)
    # cv2.rectangle(mask, gripper2_p1, gripper2_p2, (255,255,255), thickness=-1 , lineType=8, shift=0)
    # # cv2.imshow("GripperOc",mask)

    masker = cv2.imread("/home/dlrc/dlrc17-gdk/gripper.png",cv2.IMREAD_GRAYSCALE)
    print(masker.shape)
    print(mask.shape)
    print(mask.max())
    print(masker.max())
    print(masker.dtype)
    print(mask.dtype)
    # print(masker)
    # maskc = np.copy(mask)
    maskc = np.multiply(masker/255,mask)
    maskc = maskc.astype('uint8')

    # maskc[:,:,0] = np.multiply(masker,mask[:,:,0])
    # maskc[:,:,1] = np.multiply(masker,mask[:,:,1])
    # maskc[:,:,2] = np.multiply(masker,mask[:,:,2])
    cv2.imshow("GripperOc", maskc)
    cv2.imshow("GripperOcOrig", mask)
    cv2.waitKey(1)

    return maskc