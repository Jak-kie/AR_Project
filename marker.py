# import librerie esterne
import cv2


orb = cv2.ORB_create()

class Marker(object):
    def __init__(self):
        self._image, self._imagePts, self._imageDsc = None, None, None
        global orb

    def setImage(self, image):
        self._image = image

    def findDescriptors(self):
        # TODO: lancia una eccezione se self._image non è stata inizializzata != None
        self._imagePts = orb.detect(self._image, None)
        self._imagePts, self._imageDsc = orb.compute(self._image, self._imagePts)

    def getImage(self):
        return self._image
    
    def getImagePts(self):
        return self._imagePts
    
    def getImageDsc(self):
        return self._imageDsc
    
    def getImageFeatures(self):
        return cv2.drawKeypoints(self._image, self._imagePts, self._imageDsc, color=(0,255,0), flags=0)