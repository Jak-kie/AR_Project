# import moduli esterni
import cv2


orb = cv2.ORB_create()
bfMatch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class Marker(object):
    def __init__(self):
        self._image, self._imagePts, self._imageDsc = None, None, None
        global orb
        global bfMatch

    def setImage(self, image):
        self._image = image

    def findDescriptors(self):
        # TODO: lancia una eccezione se self._image non è stata inizializzata != None
        # self._imagePts = orb.detect(self._image, None)
        # self._imagePts, self._imageDsc = orb.compute(self._image, self._imagePts)
        self._imagePts, self._imageDsc = orb.detectAndCompute(self._image, None)

    def featureMatching(self, sourceImage):
        sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)
        matches = bfMatch.match(self._imageDsc, sourceImageDsc)
        matches = sorted(matches, key=lambda x: x.distance)
        # return sourceImagePts, matches
        return matches, sourceImagePts

    def getImage(self):
        return self._image
    
    def getImagePts(self):
        return self._imagePts
    
    def getImageDsc(self):
        return self._imageDsc
    
    def getImageFeatures(self):
        """Disegna i key points sulla immagine originale

        Returns:
            _type_: _description_
        """
        return cv2.drawKeypoints(self._image, self._imagePts, self._imageDsc, color=(0,255,0), flags=0)