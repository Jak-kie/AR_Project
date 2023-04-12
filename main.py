# import moduli esterni
import cv2
import numpy as np

# import moduli custom
from renderer import projection_matrix, render
from initialize import *

# import moduli PIP
from objloader_simple import *


def main():
    """Funzione main.
    Chiama tutti i sottopassaggi della pipeline.
    """

    print ("Inizializzazione in corso...")
    # Inizializzazione
    markerReference, cameraParameters, objDict, cameraVideo = initialize()

    if cameraVideo.isOpened(): # try to get the first frame
        isCameraActive, _ = cameraVideo.read()
    else:
        isCameraActive = False

    print ("Inizializzazione terminata!")

    while isCameraActive:
        # Lettura del frame dalla camera
        isCameraActive, cameraInput = readFromCamera(cameraVideo)
        grayInput = cv2.cvtColor(cameraInput, cv2.COLOR_BGR2GRAY)

        # Feature matching
        bestMarker, sourceImagePts, matches = featureMatching(markerReference, grayInput)
        if bestMarker != -1:
            # print("Trovato il marker" , markerReference[bestMarker].getPath() , ". Posizione nell'array =" , bestMarker)
            """
            plt.imshow(markerReference[bestMarker].getImage(), cmap='gray')
            plt.title("Marker")
            plt.show()
            """
            homography, transformedCorners = applyHomography(markerReference[bestMarker], sourceImagePts, matches)
            frame = cameraInput
            # frame = cv2.polylines(cameraInput, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA)
            """
            plt.figure(figsize=(12, 6))
            plt.imshow(frame, cmap='gray')
            plt.title("frame 1")
            plt.show()
            """
            
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(cameraParameters, homography)  

            # project cube or model
            # passato il modello associato al marker
            obj = objDict[markerReference[bestMarker].getPath()]
            frame = render(frame, obj, projection, markerReference[bestMarker].getImage(), True)

            # show result
            cv2.imshow('preview', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            # print ("Nessun marker trovato")
            cv2.imshow('preview', cameraInput)
            
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    
    cameraVideo.release()
    cv2.destroyWindow("preview")


def readFromCamera(cameraVideo):
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- immagine presa dalla camera
    """

    """
    Risultati in base alle immagini
            matches_01      matches_02      matches_03
    img 1
    1	        126	            145	            150
    2	        123	            159	            159
    3	        128	            162	            157
    4	        115	            146	            149

    img 2
    1	        91	            158	            97
    2	        78	            162	            125
    3	        92	            177	            113
    4	        79	            172	            101

    img 3
    1	        119	            150	            206
    2	        142	            150	            210
    3	        118	            151	            168
    4	        134	            138	            172    
    
    noMarker	104	            112	            112

    CONCLUSIONE: matches_01 troppo basso --> ARmarker_01 e simili non sono accettabili, troppi pattern ripetuti
        Gli altri 2 vanno bene. MIN_MATCHES deve stare tra 112-158 AND 112-172
    """

    # carico dalla cartella
    # imagePath = "pictures\sourceImage_02_01.jpg"
    # print (cv2.imread(imagePath, 0).shape)
    # print (cv2.imread(imagePath, 0))
    # return cv2.imread(imagePath, 0)

    isCameraActive, frame = cameraVideo.read()
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
    # numeri provati: 135, 120
    MIN_MATCHES = 125
    # numero di matches piu alti trovati
    currentBestAmount = 0
    # indice in markerReference
    currentBestMarker = -1
    currentBestMatches, currentBestSourceImagePts = None, None
    for index, entry in enumerate(markerReference):
        matches, sourceImagePts = entry.featureMatching(sourceImage)
        # sourceImagePts, sourceImageDsc, matches = entry.featureMatching(sourceImage)
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