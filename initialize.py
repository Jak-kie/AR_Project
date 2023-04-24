"""
Qua vengono effettuate le operazioni all'avvio, una volta soltanto.
Usiamo un altro file solo per comodità.
"""

# import moduli esterni
import cv2
import numpy as np
import argparse
import sys

# import moduli custom
from marker import Marker

# import moduli PIP
from objloader_simple import *


def loadReferenceMarker():
    """Genera i descriptor delle reference image
    Va effettuato solo all'avvio

    Returns:
        list -- array di marker delle reference
    """
        
    markerReference = []
    with os.scandir("pictures\marker") as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                # print(entry.name)
                tmpMarker = Marker()
                # In the case of color images, the decoded images will have the channels stored in B G R order.
                tmpMarker.setPath(entry.path)
                tmpMarker.setImage(cv2.imread(entry.path, 0))
                tmpMarker.findDescriptors()
                markerReference.append(tmpMarker)
    return markerReference


def initialize():
    # Feature detection/description dei marker reference
    # markerReference = loadReferenceMarker()

    # TODO: capire il perche di sti valori per la camera
    # camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    camera_parameters = camera_parameters.astype(float)                 # conversione necessaria per cv2.drawFrameAxes

    # Dictionary dei modelli 3D: key="NOME_FILE.jpg"  value=list["obj caricato", scalingScale dell'obj]
    # e.g. key = ARmarker_01    value=OBJ("...", swapyz=True)
    # TODO: pensare ad un modo per farli associare dall'utente, per ora hardcodiamoli cosi
    objDict = {}
    objDict.update({"pictures\marker\ARmarker_03.jpg": [OBJ("models\low-poly-fox\low-poly-fox.obj", swapyz=True), 100]})    
    objDict.update({"pictures\marker\ARmarker_04.jpg": [OBJ("models\sign-post\sign-post.obj", swapyz=True), 1000]})

    # Inizializzo la camera
    cameraVideo = cv2.VideoCapture(0)

    arucoDict, arucoParams = loadAruco()

    return camera_parameters, objDict, cameraVideo, arucoDict, arucoParams


def loadAruco():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
        default="DICT_6X6_1000",
        help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())
    
    # define names of each possible ArUco tag OpenCV supports with a dictionary
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit()

    print("[INFO] detecting '{}' tags...".format(args["type"]))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters_create()

    print(args)
    arucoType = ARUCO_DICT[args["type"]]
    print(arucoType)

    return arucoDict, arucoParams    