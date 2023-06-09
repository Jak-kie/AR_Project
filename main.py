# sviluppo tramite ArucoMarkers integrato con OpenGL e PyGame
"""
L'approccio che cerchiamo deve permettere le seguenti features:
    - caricamento modello con texture
    - utilizzo diretto della view matrix per visualizzare il modello orientato correttamente
    - setting di un background come sfondo
    Tutto cio preferibilmente senza complicarmi la vita in maniera indicibile.
    Miglior candidato finora? PyGame con OpenGL
"""

# import moduli esterni
import cv2
import numpy as np
import glob
import sys

# import moduli custom
from initialize import *
from imageLoader import ImageLoader
from objloader import *

# per testare quando tempo ci mette ad una esecuzione
# from time import process_time

# per pygame
import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


# VARIABILI GLOBALI
INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                        [-1.0,-1.0,-1.0,-1.0],
                        [-1.0,-1.0,-1.0,-1.0],
                        [ 1.0, 1.0, 1.0, 1.0]])

# per capire se la camera è calibrata. all'inizio e/o quando cambiamo lo zoom è False, è True negli altri casi
# e quindi cameraMatrix e distCoeff sono corretti
# INVERSE_MATRIX è giusto per assegnare qualche valore, non è essenziale
isCameraCalibrated = False
cameraMatrix = INVERSE_MATRIX
distCoeff = INVERSE_MATRIX


# def detect(cameraMatrix, cameraVideo, arucoDict, arucoParams):
def detect(cameraVideo, arucoDict, arucoParams):
    global INVERSE_MATRIX
    global cameraMatrix
    global distCoeff
    global isCameraCalibrated

    """
    if cameraVideo.isOpened(): # try to get the first frame
        isCameraActive, _ = cameraVideo.read()
    else:
        isCameraActive = False
        sys.exit()
    """
    # Lettura del frame dalla camera
    rgbInput = readFromCamera(cameraVideo)
    grayInput = cv2.cvtColor(rgbInput, cv2.COLOR_BGR2GRAY)
    arucoCorners, arucoIds = arucoMatching(grayInput, arucoDict, arucoParams)
    if (len(arucoCorners) > 0):
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
            """
            # posizioni 2D degli angoli dei marker
            floatCorners = (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))

            """
            # testing per verificare come bene rileva il marker
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
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
            """

            # Pose Estimation and Camera calibration
            markerLength = 0.10
            if not isCameraCalibrated:
                estimateCameraParameters(floatCorners)
                isCameraCalibrated = True
            # print ("cameraMatrix: " , cameraMatrix)
            # print ("distCoeff: " , distCoeff)
            rotVecs, transVecs = estimatePoseSingleMarkers(floatCorners, markerLength)
            # print ("rotVecs: " , rotVecs)
            # print ("transVecs: " , transVecs)

            R = cv2.Rodrigues(rotVecs)[0]
            T = transVecs
            """view_matrix = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
                [R[1,0],R[1,1],R[1,2],T[1]],
                [R[2,0],R[2,1],R[2,2],T[2]],
                [0.0, 0.0, 0.0, 1.0]])"""
            view_matrix = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
                                    [R[1,0],R[1,1],R[1,2],T[1]],
                                    [R[2,0],R[2,1],R[2,2],T[2]],
                                    [ 0.0,   0.0,   0.0,   1.0]],
                                    np.float32)
            # print ("view_matrix: " , view_matrix)               # camera transformation matrix
            view_matrix = view_matrix * INVERSE_MATRIX
            # print ("view_matrix * INVERSE_MATRIX: " , view_matrix)
            view_matrix = np.transpose(view_matrix)
            # print ("view_matrix transpose: " , view_matrix)
            frameOutput = rgbInput
            print ("---> MARKER TROVATO")
            return frameOutput, view_matrix, True, arucoIds[0]
    else:
        print ("---> MARKER NON TROVATO")
        frameOutput = rgbInput
        # ritornamo INVERSE_MATRIX, anche se in realta non verrà usato, giusto per tornare qualcosa
        return frameOutput, INVERSE_MATRIX, False, -1
    # cv2.imshow('preview', frameOutput)


# questa funzione va chiamata quando: prima volta che troviamo il marker/risettiamo zoom della camera
# anche se puo ritornare rotVecs e transVecs, limitiamoci ai parametri intrinsechi e passiamoli a solvePnp dopo
def estimateCameraParameters(corners):
    global cameraMatrix
    global distCoeff

    # CAMERA CALIBRATION
    # https://docs.opencv.org/4.6.0/dc/dbb/tutorial_py_calibration.html
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.    

    images = glob.glob('pictures\calibrateCamera\*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    _ , cameraMatrix, distCoeff, _ , _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # DEBUG TEMPORANEO    
    # cameraMatrix = np.array([[100, 0, 30], [0, 100, 30], [0, 0, 1]])
    # cameraMatrix = cameraMatrix.astype(float)         # conversione necessaria per cv2.drawFrameAxes
    # distCoeff = np.array([0, 0.5, 1.0, 1.0])


def estimatePoseSingleMarkers(corners, marker_size):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    '''
    global cameraMatrix
    global distCoeff
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
        nada, R, t = cv2.solvePnP(marker_points, corners[i], cameraMatrix, distCoeff, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs
    """
    _, rotVecs, transVecs = cv2.solvePnP(marker_points, corners, cameraMatrix, distCoeff, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return rotVecs, transVecs


def readFromCamera(cameraVideo):
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- immagine presa dalla camera
    """

    _ , frame = cameraVideo.read()
    # frame = cv2.imread("pictures\\arucoMarker_5x5.jpg")
    return frame


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

    """
        corners (tuple)
        ids --> se trovato: (numpy.ndarray)
            --> se non trovato: (NoneType)
    """
    (corners, ids, _) = cv2.aruco.detectMarkers(image, dict, parameters=params)
    return corners, ids


def main():
    # Inizializzazione
    print ("Inizializzazione in corso...")

    # (0,0) indica l'offset dal top-left angolo della viewbox. con (0,0) non c'è offset
    im_loader = ImageLoader(0, 0)

    # Inizializzazione pygame
    pg.init()
    pgClock = pg.time.Clock()
    width = 640
    height = 480
    displayRes = (width, height)
    FLAGS = DOUBLEBUF | OPENGL
    gameDisplay = pg.display.set_mode(displayRes, FLAGS)

    # Inizializzazione parametri + dizionario modelli
    objDict, cameraVideo, arucoDict, arucoParams = initialize()

    print ("Inizializzazione terminata!")

    print ("1. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
    print ("2. glGetFloatv GL_PROJECTION_MATRIX:" , glGetFloatv(GL_PROJECTION_MATRIX))

    # firstTime = True
    running = True
    while running:
        # pg.time.wait(10)
        pgClock.tick(60)

        # gestore di eventi
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                print ("QUIT: closing pygame...")
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    print ("ESCAPE: closing pygame...")

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # ottieni il frame
        frame, view_matrix, retval, arucoIds = detect(cameraVideo, arucoDict, arucoParams)

        # render del background
        # per elementi 2D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        print ("3. glGetFloatv GL_PROJECTION_MATRIX:" , glGetFloatv(GL_PROJECTION_MATRIX))
        # defnisce la 2-D orthographic projection matrix \ viewbox: (left, right, top, bottom) \ (left, right, bottom, top)
        # no perspective
        # https://stackoverflow.com/questions/1401326/gluperspective-vs-gluortho2d
        gluOrtho2D(0, width, height, 0)
        print ("4. glGetFloatv GL_PROJECTION_MATRIX:" , glGetFloatv(GL_PROJECTION_MATRIX))
        glMatrixMode(GL_MODELVIEW)
        # glPushMatrix()
        glLoadIdentity()
        print ("5. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))

        glDisable(GL_DEPTH_TEST)            # non fa controlli per il drawing di pixel nascosti
        im_loader.load(frame)
        glColor3f(1, 1, 1)                  # va tenuto per non avere lo sfondo colorato diversamente
        im_loader.draw()
        print ("7. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
        # glPopMatrix()

        if retval:
            # recupero delle info sul modello
            print("arucoIds: " , arucoIds)
            obj = objDict[arucoIds][0]
            scalingScale = objDict[arucoIds][1]
            # render del modello
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            # vedi il link sopra
            gluPerspective(90, (displayRes[0]/displayRes[1]), 0.1, 50.0)
            print ("8. glGetFloatv GL_PROJECTION_MATRIX:" , glGetFloatv(GL_PROJECTION_MATRIX))

            glMatrixMode(GL_MODELVIEW)
            # !!!!!! FARE glLoadIdentity() + glMultMatrixd(a) == glLoadMatrixd(a)
            # glLoadIdentity()
            # glMultMatrixd(view_matrix)
            # print ("9. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
            glLoadMatrixd(view_matrix)
            print ("10. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
            # scaling (N.B. potrebbe essere valido solo per alcuni modelli, potremmo doverlo settare diversamente per ciascuno)
            glScalef(scalingScale, scalingScale, scalingScale)
            # glRotatef(90, 1, 0, 0)                # commentanto perche carichiamo il modello con swapyz=True
            print ("11. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
            # glRotatef(1, 0, 1, 0)
            # print ("10. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))

            # voglio effettuare la traslazione SOLO LA PRIMA VOLTA, non ad ogni frame
            # if firstTime:
                # glTranslatef(0.0, -1, -7)                   # per la fox
                # glTranslatef(0.0, -10, -50)               # per il tie-fighter
                # print ("11. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
                # firstTime = False

            # rendering del modello
            glEnable(GL_DEPTH_TEST)
            # glEnable(GL_DEPTH_TEST | GL_NORMALIZE)
            obj.render()

        # switch del buffer
        pg.display.flip()

        # print ("rendering in corso...")

    print ("Rendering terminato.")
    sys.exit(1) # quit()


if __name__ == '__main__':
    main()