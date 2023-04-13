"""
Qua vengono effettuate le operazioni all'avvio, una volta soltanto.
Usiamo un altro file solo per comodità.
"""

# import moduli esterni
import cv2
import numpy as np

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
    markerReference = loadReferenceMarker()

    # TODO: capire il perche di sti valori per la camera
    # camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # Dictionary dei modelli 3D: key="NOME_FILE.jpg"  value=list["obj caricato", scalingScale dell'obj]
    # e.g. key = ARmarker_01    value=OBJ("...", swapyz=True)
    # TODO: pensare ad un modo per farli associare dall'utente, per ora hardcodiamoli cosi
    objDict = {}
    objDict.update({"pictures\marker\ARmarker_03.jpg": [OBJ("models\low-poly-fox\low-poly-fox.obj", swapyz=True), 100]})    
    objDict.update({"pictures\marker\ARmarker_04.jpg": [OBJ("models\sign-post\sign-post.obj", swapyz=True), 1000]})

    # Inizializzo la camera
    cameraVideo = cv2.VideoCapture(0)

    return markerReference, camera_parameters, objDict, cameraVideo