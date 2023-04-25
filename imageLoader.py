import cv2
from PIL import Image

from OpenGL.GL import *
from OpenGL.GLU import *


class ImageLoader:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 0
        self.height = 0
        self.img_data = 0


    def load(self, image):
        im = image
        tx_image = cv2.flip(im, 0)
        tx_image = Image.fromarray(tx_image)
        self.width = tx_image.size[0]
        self.height = tx_image.size[1]
        self.img_data = tx_image.tobytes('raw', 'BGRX', 0, -1)

        self.Texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.Texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img_data)


    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(self.x, self.y, 0)
        
        glEnable(GL_TEXTURE_2D)  
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(self.width, 0)
        glTexCoord2f(1, 1)
        glVertex2f(self.width, self.height)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.height)
        glEnd()
        glDisable(GL_TEXTURE_2D)