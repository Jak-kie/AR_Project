# import librerie esterne
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

#import moduli custom
import recognition as rcn
from marker import Marker


def main():
    """Funzione main.
    Chiama tutti i sottopassaggi della pipeline.
    """

    markerReference = descriptorReference()

    # cameraInput = readFromCamera()


def readFromCamera():
    """Riceve la immagine dalla camera del dispositivo.
    Per ora la immagine la otteniamo tramite lettura da file.

    Returns:
        ndarray -- Immagine presa dalla camera
    """

    imagePath = "D:\Marco\immagini\jj.jpg"
    return plt.imread(imagePath)


def descriptorReference():
    """Genera i descriptor delle reference image
    Va effettuato solo all'avvio
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

if __name__ == '__main__':
    main()