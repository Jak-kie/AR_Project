# sviluppo tramite ArucoMarkers integrato con OpenGL

# import moduli esterni
import cv2
import numpy as np

# import moduli custom
from renderer import projection_matrix, render, renderV2
from initialize import *

# import moduli PIP
from objloader_simple import *

# per testare quando tempo ci mette ad una esecuzione
from time import process_time

# per aruco markers
import argparse
import sys


def main():
    """Funzione main.
    Chiama tutti i sottopassaggi della pipeline.
    """

    print ("Inizializzazione in corso...")
    # Inizializzazione
    cameraParameters, _, cameraVideo, arucoDict, arucoParams = initialize()

    if cameraVideo.isOpened(): # try to get the first frame
        isCameraActive, _ = cameraVideo.read()
    else:
        isCameraActive = False
        sys.exit()

    print ("Inizializzazione terminata!")

    while isCameraActive:
        
        print ("-------------------------------------------------------------------------")

        startTimeFinal = process_time()

        # Lettura del frame dalla camera
        isCameraActive, rgbInput = readFromCamera(cameraVideo)
        grayInput = cv2.cvtColor(rgbInput, cv2.COLOR_BGR2GRAY)

        # tempo bassissimo, 0.0 seconds
        endTimeGetGray = process_time()
        print("Tempo per leggere il frame e convertirlo in grayscale --- %s seconds ---" % (endTimeGetGray - startTimeFinal))        

        startTimeMatching = process_time()
        arucoCorners, arucoIds = arucoMatching(grayInput, arucoDict, arucoParams)
        endTimeMatching = process_time()
        print("Tempo per arucoMatching --- %s seconds ---" % (endTimeMatching - startTimeMatching))       

        if (len(arucoCorners) > 0):
            print ("MARKER TROVATO")
            arucoIds = arucoIds.flatten()

            # TODO: gestire caso in cui ci sono piu marker, per ora limitiamoci a 1 solo nella scena
            for (markerCorner, markerID) in zip(arucoCorners, arucoIds):
                """
                arucoCorner è costituito dalle posizioni in ordine orario
                TL --> TR
                        |
                        |
                        V
                BL <-- BR

                esempio di print:
                markerCorner:  [[[215. 107.]
                    [388.  83.]
                    [397. 273.]
                    [218. 282.]]]
                # pre int 
                topLeft:  [215. 107.]
                topRight:  [388.  83.]
                bottomRight:  [397. 273.]
                bottomLeft:  [218. 282.]
                # post int
                topLeft:  (215, 107)
                topRight:  (388, 83)
                bottomRight:  (397, 273)
                bottomLeft:  (218, 282)
                """
                # posizioni 2D degli angoli dei marker
                floatCorners = (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
                
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                # testing per verificare come bene rileva il marker
                # disegniamo la linea che delimita il marker
                cv2.line(rgbInput, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(rgbInput, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(rgbInput, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(rgbInput, bottomLeft, topLeft, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(rgbInput, (cX, cY), 4, (0, 0, 255), -1)
		        
                # draw the ArUco marker ID on the frame
                cv2.putText(rgbInput, str(markerID), (topRight[0], topRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Pose Estimation
                # objp = np.array([[0.,0.,0.],[1.,0.,0.], [1.,1.,0.],[0.,1.,0.]], dtype='float32')
                dist = np.zeros((4,1))          # temporaneo
                markerLength = 0.10
                startTimeEstimate = process_time()
                rotVecs, transVecs = my_estimatePoseSingleMarkers(floatCorners, markerLength, cameraParameters, dist)
                endTimeEstimate = process_time()
                print("Tempo per estimate --- %s seconds ---" % (endTimeEstimate - startTimeEstimate))

                """
                print ("rotVecs: " , rotVecs)
                print ("transVecs: " , transVecs)
                print ("rotVecs shape: " , rotVecs.shape)
                print ("transVecs shape: " , transVecs.shape)
                print ("cameraParameters: ", cameraParameters)
                print ("cameraParameters shape: ", cameraParameters.shape)            
                rgbInput = cv2.drawFrameAxes(rgbInput, cameraParameters, dist, rotVecs, transVecs, markerLength)
                """

                # get homography matrix
                K = cameraParameters
                D = dist
                R = cv2.Rodrigues(rotVecs)[0]
                T = transVecs

                print ("K: " , K)
                print ("D: " , D)
                print ("R: " , R)
                print ("T: " , T)

                # debug
                """
                R2 = cv2.Rodrigues(rotVecs)
                R3 = cv2.Rodrigues(rotVecs)[1]
                print ("R2: " , R2)
                print ("R3: " , R3)
                """

                INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [ 1.0, 1.0, 1.0, 1.0]])
                """view_matrix = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
                    [R[1,0],R[1,1],R[1,2],T[1]],
                    [R[2,0],R[2,1],R[2,2],T[2]],
                    [0.0, 0.0, 0.0, 1.0]])"""
                view_matrix = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
                                        [R[1,0],R[1,1],R[1,2],T[1]],
                                        [R[2,0],R[2,1],R[2,2],T[2]],
                                        [  0.0,   0.0,   0.0,   1.0]],
                                        np.float32)
                print ("view_matrix: " , view_matrix)               # camera transformation matrix
                view_matrix = view_matrix * INVERSE_MATRIX
                print ("view_matrix * INVERSE_MATRIX: " , view_matrix)
                view_matrix = np.transpose(view_matrix)
                print ("view_matrix transpose: " , view_matrix)
                # projection = projection_matrix(cameraParameters, view_matrix)  

                # rendering
                # obj = objDict[markerReference[0].getPath()]
                # TODO: controlla lo swapyz
                obj = [OBJ("models\low-poly-fox\low-poly-fox.obj", swapyz=True), 100]
                frameOutput = rgbInput

                # frameOutput = renderV2(rgbInput, obj[0], view_matrix, obj[1], color=False)

                # frame = render(frame, obj[0], projection, markerReference[bestMarker].getImage(), obj[1], True)

        else:
            print ("MARKER NON TROVATO")
            frameOutput = rgbInput

        # decommentare se voglio esaminare un solo frame
        # isCameraActive = False

        # cv2.imshow('preview', rgbInput)
        cv2.imshow('preview', frameOutput)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    
    cameraVideo.release()
    cv2.destroyWindow("preview")


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    """
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs
    """
    _, rotVecs, transVecs = cv2.solvePnP(marker_points, corners, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return rotVecs, transVecs


def readFromCamera(cameraVideo):
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- immagine presa dalla camera
    """

    isCameraActive, frame = cameraVideo.read()
    # frame = cv2.imread("pictures\\arucoMarker_5x5.jpg")
    # print (frame.shape)
    # print (frame)
    return isCameraActive, frame


def featureMatching(markerReference, sourceImage):
    """Verifica se (e quale) marker è presente nella scena.
    Riporta l'indice associato al marker in markerReference
    se è trovato, -1 se non è presente alcun marker.

    Returns:
        int
    """

    # minimo ammontare di matchesAmount perche venga considerato valido
    # numeri provati: 135, 120, 125, 140, 200, 175
    MIN_MATCHES = 140
    # numero di matches piu alti trovati
    currentBestAmount = 0
    # indice in markerReference
    currentBestMarker = -1
    currentBestMatches, currentBestSourceImagePts = None, None
    for index, entry in enumerate(markerReference):
        # sourceImagePts, sourceImageDsc, matches = entry.featureMatching(sourceImage)
        matches, sourceImagePts = entry.featureMatching(sourceImage)
        # print ("Path del marker:" , entry.getPath())
        # print ("Numero di matches:" , len(matches))
        if len(matches) > MIN_MATCHES and len(matches) > currentBestAmount: 
            currentBestAmount = len(matches)
            currentBestMarker = index
            currentBestMatches = matches
            currentBestSourceImagePts = sourceImagePts
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
    return currentBestMarker, currentBestSourceImagePts, currentBestMatches


def arucoMatching(image, dict, params):
    """detect degli Aruco Marker nella immagine

    Args:
        image (ndarray): immagine grayscale dell'input camera
        dict (_type_): _description_
        params (_type_): _description_
    """

    # prepariamo la immagine per l'analisi
    # TODO: vedere se queste preparazioni migliorano il rilevamento del marker
    # image = cv2.GaussianBlur(image, (5,5), 0)
    # edges = cv2.Canny(gray, 100, 200)

    # detection con aruco
    """
        corners (tuple)
        ids --> se trovato: (numpy.ndarray)
            --> se non trovato: (NoneType)
    """
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, dict, parameters=params)

    # cv2.imshow('preview', image)

    """
    if (len(corners) > 0):
        print ("MARKER TROVATO")
        print ("corners: " , corners)
        print ("ids: " , ids)
    else:
        pass
        print ("MARKER NON TROVATO")
        # print ("rejected:" , rejected)
    """

    return corners, ids


def applyHomography(marker, sourceImagePts, matches):
    # Get the good key points positions
    sourcePoints = np.float32([marker.getImagePts()[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Obtain the homography matrix
    homography, _ = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
    return homography


if __name__ == '__main__':
    main()