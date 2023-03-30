# import librerie esterne
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

#import moduli custom
import marker as mk


def descriptorReference():
    """Genera i descriptor delle reference image
    Va effettuato solo all'avvio
    """