# import moduli esterni
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# import moduli custom
from marker import Marker
from renderer import projection_matrix, render

# import moduli PIP
"""
import pygame
from pygame.locals import *
from pygame.constants import *
from objloader import *
from OpenGL.GL import *
from OpenGL.GLU import *
"""
from objloader_simple import *


def main():
    """Funzione main.
    Chiama tutti i sottopassaggi della pipeline.
    """

    # Feature detection/description dei marker reference
    markerReference = descriptorReference()

    camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

    # Caricamento modello, per ora carichiamolo direttamente 1 volta sola, poi andra caricato in base al marker
    """
    pygame.init()
    viewport = (800,600)
    hx = viewport[0]/2
    hy = viewport[1]/2
    pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
    """
    obj = OBJ("models\headcrab-obj\headcrab.obj", swapyz=True)

    # Lettura del frame dalla camera
    cameraInput = readFromCamera()

    # Feature matching
    bestMarker, sourceImagePts, matches = featureMatching(markerReference, cameraInput)
    if bestMarker != -1:
        print("Marker trovato, posizione nell'array" , bestMarker)
        """
        plt.imshow(markerReference[bestMarker].getImage(), cmap='gray')
        plt.title("Marker")
        plt.show()
        """
        homography, transformedCorners = applyHomography(markerReference[bestMarker], sourceImagePts, matches)
        frame = cv2.polylines(cameraInput, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA)

        plt.figure(figsize=(12, 6))
        plt.imshow(frame, cmap='gray')
        plt.title("frame 1")
        plt.show()

        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(camera_parameters, homography)  
        
        print(projection)

        # project cube or model
        frame = render(frame, obj, projection, markerReference[bestMarker].getImage(), False)

        plt.figure(figsize=(12, 6))
        plt.imshow(frame, cmap='gray')
        plt.title("final frame")
        plt.show()
        """
        # Draw a polygon on the second image joining the transformed corners
        
        # Draw the matches
        drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None,
                            matchesMask=matchesMask, flags=2)
        result = cv2.drawMatches(markerReference[bestMarker].getImage(), markerReference[bestMarker].getImagePts()
            , cameraInput, sourceImagePts, matches, None, **drawParameters)

        # Show image
        plt.figure(figsize=(12, 6))
        plt.imshow(result, cmap='gray')
        plt.show()
        """

    else:
        print ("Nessun marker trovato")


def readFromCamera():
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- immagine presa dalla camera
    """

    imagePath = "pictures\sourceImage_04.jpg"
    # plt.imshow(cv2.imread(imagePath, 0), cmap='gray')
    # plt.title("cameraInput")
    # plt.show()
    return cv2.imread(imagePath, 0)
    # return plt.imread(imagePath)


def descriptorReference():
    """Genera i descriptor delle reference image
    Va effettuato solo all'avvio

    Returns:
        list -- array di marker delle reference
    """
    
    # TODO: da fare per ogni marker dentro /pictures/markers   
    markerReference = []
    with os.scandir("pictures\marker") as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                # print(entry.name)
                tmpMarker = Marker()
                # In the case of color images, the decoded images will have the channels stored in B G R order.
                tmpMarker.setImage(cv2.imread(entry.path, 0))
                tmpMarker.findDescriptors()
                markerReference.append(tmpMarker)
    """plt.imshow(markerReference[0].getImage(), cmap='gray')
    plt.show()
    print(markerReference[0].getImage())
    print(markerReference[0].getImagePts())
    print(markerReference[0].getImageDsc())
    print(np.array(markerReference[0].getImage()).shape)
    
    plt.imshow(markerReference[0].getImageFeatures(), cmap='gray')
    plt.title('Reference Image Features')
    plt.show()"""

    return markerReference


def featureMatching(markerReference, sourceImage):
    """Verifica se (e quale) marker è presente nella scena.
    Riporta l'indice associato al marker in markerReference
    se è trovato, -1 se non è presente alcun marker.

    Returns:
        int
    """

    # minimo ammontare di matchesAmount perche venga considerato valido
    MIN_MATCHES = 110
    # matchesAmount migliore trovato finora
    currentBestAmount = 0
    # indice in markerReference
    currentBestMarker = -1
    for index, entry in enumerate(markerReference):
        matches, sourceImagePts = entry.featureMatching(sourceImage)
        # sourceImagePts, sourceImageDsc, matches = entry.featureMatching(sourceImage)
        print ("Numero di matches:" , len(matches))
        if len(matches) > MIN_MATCHES and len(matches) > currentBestAmount: 
            currentBestAmount = len(matches)
            currentBestMarker = index
        """
            # OUTPUT DI PROVA
            # draw first 15 matches.
            idxPairs = cv2.drawMatches(entry.getImage(), entry.getImagePts(), sourceImage, sourceImagePts, matches[:MIN_MATCHES], 0, flags=2)
            # show result
            plt.figure(figsize=(12, 6))
            plt.axis('off')
            plt.imshow(idxPairs, cmap='gray')
            plt.title('Matching between features')
            plt.show()
        else:
            print("Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES))
            matchesMask = None
        """
    return currentBestMarker, sourceImagePts, matches


def applyHomography(marker, sourceImagePts, matches):
    # Get the good key points positions
    sourcePoints = np.float32([marker.getImagePts()[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Obtain the homography matrix
    homography, _ = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
    # matchesMask = mask.ravel().tolist()

    # Apply the perspective transformation to the source image corners
    h, w = marker.getImage().shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, homography)

    return homography, transformedCorners



if __name__ == '__main__':
    main()