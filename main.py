# sviluppo tramite ArucoMarkers integrato con OpenGL
"""
TODO l'approccio che cerchiamo deve permettere le seguenti features:
    - caricamento modello con texture
    - utilizzo diretto della view matrix per visualizzare il modello orientato correttamente
    - setting di un background come sfondo
    Tutto cio preferibilmente senza complicarmi la vita in maniera indicibile.
    Miglior candidato finora? PyGame con OpenGL
"""

# import moduli esterni
import cv2
import numpy as np

# import moduli custom
from renderer import projection_matrix, render, renderV2
from initialize import *
from imageLoader import ImageLoader

# import moduli PIP
# from objloader_simple import *
from objloader import *

# per testare quando tempo ci mette ad una esecuzione
from time import process_time

# per aruco markers
import argparse
import sys

# per pygame
import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


def detectRender(cameraParameters, cameraVideo, arucoDict, arucoParams):
    if cameraVideo.isOpened(): # try to get the first frame
        isCameraActive, _ = cameraVideo.read()
    else:
        isCameraActive = False
        sys.exit()
    # Lettura del frame dalla camera
    isCameraActive, rgbInput = readFromCamera(cameraVideo)
    grayInput = cv2.cvtColor(rgbInput, cv2.COLOR_BGR2GRAY)
    arucoCorners, arucoIds = arucoMatching(grayInput, arucoDict, arucoParams)
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
            rotVecs, transVecs = my_estimatePoseSingleMarkers(floatCorners, markerLength, cameraParameters, dist)
            # get homography matrix
            K = cameraParameters
            D = dist
            R = cv2.Rodrigues(rotVecs)[0]
            T = transVecs
            """
            print ("K: " , K)
            print ("D: " , D)
            print ("R: " , R)
            print ("T: " , T)
            """
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
            # print ("view_matrix: " , view_matrix)               # camera transformation matrix
            view_matrix = view_matrix * INVERSE_MATRIX
            # print ("view_matrix * INVERSE_MATRIX: " , view_matrix)
            view_matrix = np.transpose(view_matrix)
            # print ("view_matrix transpose: " , view_matrix)
            # projection = projection_matrix(cameraParameters, view_matrix)  
            # rendering
            # obj = objDict[markerReference[0].getPath()]
            # TODO: controlla lo swapyz
            # obj = [OBJ("models\low-poly-fox\low-poly-fox.obj", swapyz=True), 100]
            frameOutput = rgbInput
            # frameOutput = renderV2(rgbInput, obj[0], view_matrix, obj[1], color=False)
            # frame = render(frame, obj[0], projection, markerReference[bestMarker].getImage(), obj[1], True)
    else:
        print ("MARKER NON TROVATO")
        frameOutput = rgbInput
    # cv2.imshow('preview', frameOutput)
    return frameOutput


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
    return corners, ids


cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))

def renderSolidCube():
    glBegin(GL_QUADS)
    for cubeQuad in cubeQuads:
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()


def main():

    # initialize detectRender
    print ("Inizializzazione in corso...")
    cameraParameters, _, cameraVideo, arucoDict, arucoParams = initialize()
    print ("Inizializzazione terminata!")
    
    im_loader = ImageLoader(0, 0)

    pg.init()

    pgClock = pg.time.Clock()

    width = 640
    height = 480
    displayRes = (width, height)
    FLAGS = DOUBLEBUF | OPENGL
    gameDisplay = pg.display.set_mode(displayRes, FLAGS)

    fox = OBJ("models\low-poly-fox\low-poly-fox.obj")
    # tieFighter = OBJ("models\star-wars-vader-tie-fighter-obj\star-wars-vader-tie-fighter.obj")
    # gameDisplay = pg.display.set_mode(displayRes)

    # setta il colore
    # colorBG = pg.Color(0, 0, 255)
    # colorBG = (0, 0, 255)

    # set background
    # background = pg.Surface(displayRes)
    # background.fill(colorBG)
    # pg.draw.rect(background,(0,255,255),(20,20,40,40))

    gluPerspective(45, (displayRes[0]/displayRes[1]), 0.1, 50.0)

    # maggiore il valore, maggiormente viene spostato
    # print("current matrix mode: " , glGetIntegerv(GL_MATRIX_MODE))
    # 5888 = GL_MODELVIEW

    # print ("1. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
    glTranslatef(0.0, -1, -7)

    print ("glGetFloatv GL_PROJECTION_MATRIX:" , glGetFloatv(GL_PROJECTION_MATRIX))
    print ("2. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))

    # angle = 0

    running = True
    while running:
        # pg.time.wait(10)
        pgClock.tick(60)

        # gestore di eventi
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                print ("QUIT: closing pygame...")
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    print ("Escape: closing pygame...")

        # ottieni il frame
        frame = detectRender(cameraParameters, cameraVideo, arucoDict, arucoParams)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        """
        # per elementi 2D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, height, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)            # non fa controlli per il drawing di pixel nascosti
        im_loader.load(frame)
        glColor3f(1, 1, 1)
        im_loader.draw()
        """

        # reset delle matrici 
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluPerspective(45, (displayRes[0]/displayRes[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        print ("3. glGetFloatv GL_MODELVIEW_MATRIX:" , glGetFloatv(GL_MODELVIEW_MATRIX))
        # glLoadIdentity()
        # glTranslate(0.0, 0.0, -5)
        # glRotate(angle, 0, 1, 0)          # gira sempre piu velocemente
        # angle += 1
        # glTranslate(0.0, 0.0, -0.05)
        glRotatef(1, 0, 1, 0)
        print ("4. glGetFloatv GL_MODELVIEW_MATRIX POST ROTATE:" , glGetFloatv(GL_MODELVIEW_MATRIX))

        # glRotatef(1, 1, 1, 1)

        # rendering effettivo
        glEnable(GL_DEPTH_TEST)
        fox.render()
        # tieFighter.render()
        # renderSolidCube()

        # usiamo il buffer corretto
        pg.display.flip()

        """
        # alternativa 1
        pg.surfarray.blit_array(background, frame)
        # bkgr = pg.image.load(frame)
        # bkgr.convert()
        gameDisplay.blit(background, (0,0))
        """
        # alternativa 2
        # background = pg.surfarray.make_surface(frame)
        # gameDisplay.blit(background, (0,0))

        # render degli oggetti nella scena
        # renderSolidCube()

        # pg.display.flip()

        print ("rendering in corso...")

    # pg.quit()
    print ("rendering terminato")
    sys.exit(1) # quit()


# questa versione NON usa OpenGL
def main_OLD():

    # initialize detectRender
    print ("Inizializzazione in corso...")
    cameraParameters, _, cameraVideo, arucoDict, arucoParams = initialize()
    print ("Inizializzazione terminata!")

    pg.init()

    pgClock = pg.time.Clock()

    width = 640
    height = 480
    displayRes = (width, height)
    FLAGS = DOUBLEBUF | OPENGL
    # gameDisplay = pg.display.set_mode(displayRes, FLAGS)
    gameDisplay = pg.display.set_mode(displayRes)

    # setta il colore
    # colorBG = pg.Color(0, 0, 255)
    # colorBG = (0, 0, 255)

    # set background    
    background = pg.Surface(displayRes)
    # background.fill(colorBG)
    # pg.draw.rect(background,(0,255,255),(20,20,40,40))

    # gluPerspective(60, (displayRes[0]/displayRes[1]), 0.1, 100.0)

    # glTranslatef(0.0, 0.0, -5)

    running = True
    while running:
        pg.time.wait(10)
        # pgClock.tick(60)

        # gestore di eventi
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                print ("QUIT: closing pygame...")
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    print ("Escape: closing pygame...")

        # glRotatef(1, 1, 1, 1)
        # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # otteniamo la immagine di sfondo
        frame = detectRender(cameraParameters, cameraVideo, arucoDict, arucoParams)
        # le immagini sono in formato (height, width, channels) (480, 640, 3). opencv legge le immagini in BGR
        # a noi servono immagini dove width > height, (640, 480, 3), in formato RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(1, 0, 2)
        """
        # alternativa 1
        pg.surfarray.blit_array(background, frame)
        # bkgr = pg.image.load(frame)
        # bkgr.convert()
        gameDisplay.blit(background, (0,0))
        """
        # alternativa 2
        background = pg.surfarray.make_surface(frame)
        gameDisplay.blit(background, (0,0))

        # render degli oggetti nella scena
        # renderSolidCube()

        pg.display.flip()

        print ("rendering in corso...")

    # pg.quit()
    sys.exit(1) # quit()


if __name__ == '__main__':
    main()