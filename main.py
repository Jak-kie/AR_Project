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

    # Inizializzazione
    markerReference, cameraParameters, objDict = initialize()

    # Lettura del frame dalla camera
    cameraInput = readFromCamera()

    # Feature matching
    bestMarker, sourceImagePts, matches = featureMatching(markerReference, cameraInput)
    if bestMarker != -1:
        print("Trovato il marker" , markerReference[bestMarker].getPath() , ". Posizione nell'array =" , bestMarker)
        """
        plt.imshow(markerReference[bestMarker].getImage(), cmap='gray')
        plt.title("Marker")
        plt.show()
        """
        homography, transformedCorners = applyHomography(markerReference[bestMarker], sourceImagePts, matches)
        frame = cv2.polylines(cameraInput, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA)
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
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print ("Nessun marker trovato")     # TODO: FUCKING WISH IT DID WORK HOLY MOLY


def readFromCamera():
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- immagine presa dalla camera
    """

    """
    Risultati in base alle immagini
    img     matches_01      matches_02
    1	        126	            145
    2	        123	            159
    3	        128	            162
    4	        115	            146

    5	        91	            158
    6	        78	            162
    7	        92	            177
    8	        79	            172

    noMarker	104	            112
    """

    imagePath = "pictures\sourceImage_03_04.jpg"
    return cv2.imread(imagePath, 0)


def featureMatching(markerReference, sourceImage):
    """Verifica se (e quale) marker è presente nella scena.
    Riporta l'indice associato al marker in markerReference
    se è trovato, -1 se non è presente alcun marker.

    Returns:
        int
    """

    # minimo ammontare di matchesAmount perche venga considerato valido
    MIN_MATCHES = 135
    # matchesAmount migliore trovato finora
    currentBestAmount = 0
    # indice in markerReference
    currentBestMarker = -1
    currentBestMatches, currentBestSourceImagePts = None, None
    for index, entry in enumerate(markerReference):
        matches, sourceImagePts = entry.featureMatching(sourceImage)
        # sourceImagePts, sourceImageDsc, matches = entry.featureMatching(sourceImage)
        print ("Path del marker:" , entry.getPath())
        print ("Numero di matches:" , len(matches))
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