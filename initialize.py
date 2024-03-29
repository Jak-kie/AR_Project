"""
Qua vengono effettuate le operazioni all'avvio, una volta soltanto.
"""

# import moduli esterni
import cv2
import argparse
import sys

# import moduli custom
from objloader import *


def initialize(displayRes):
    """Carica il dizionario dei modelli e i parametri Aruco

    Args:
        displayRes (int[]) : tupla contenente width e height, per settare la camera
    Returns:
        objDict (dictionary) : dict dei modelli "key=id aruco marker , value=list["obj caricato", scalingScale]"
        cameraVideo (cv2.VideoCapture(0)) : reference alla camera usata
        arucoDict (cv2.aruco.Dictionary) : dizionario dei possbili marker trovabili
        arucoParams (cv2.aruco.DetectorParameters) : parametri di arucoMarker
    """

    # Inizializzo il dictionary dei modelli 3D
    print ("---> Caricamento modelli in corso...")
    objDict = {}
    objDict.update({21: [OBJ("models\low-poly-fox\low-poly-fox.obj", swapyz=True), 0.05]})
    objDict.update({151: [OBJ("models\star-wars-vader-tie-fighter-obj\star-wars-vader-tie-fighter.obj", swapyz=True), 0.004]})    
    print ("---> Caricamento modelli terminato!")

    # Inizializzo la camera
    print ("---> Inizializzazione della camera...")
    cameraVideo = cv2.VideoCapture(0)
    # setta la risoluzione della camera
    cameraVideo.set(cv2.CAP_PROP_FRAME_WIDTH, displayRes[0])
    cameraVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, displayRes[1])
    if not cameraVideo.isOpened(): # try to get the first frame
        print ("!!! ERRORE DURANTE IL CARICAMENTO DELLA CAMERA, CHIUSURA !!!")
        sys.exit()
    print ("---> Camera inizializzata!")

    # Inizializzo i parametri aruco
    print ("---> Caricamento parametri arucoMarker...")
    arucoDict, arucoParams = loadAruco()
    print ("---> Parametri arucoMarker caricati!")

    return objDict, cameraVideo, arucoDict, arucoParams


def loadAruco():
    """Carica il dizionario dei modelli e i parametri Aruco

    Args:
    Returns:
        arucoDict (cv2.aruco.Dictionary) : dizionario dei possbili marker trovabili
        arucoParams (cv2.aruco.DetectorParameters) : parametri di arucoMarker
    """

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

    # print(args)
    # arucoType = ARUCO_DICT[args["type"]]
    # print(arucoType)
    # print ("type(arucoDict)", type(arucoDict))
    # print ("type(arucoParams)", type(arucoParams))

    return arucoDict, arucoParams    