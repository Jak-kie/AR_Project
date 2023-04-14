# import moduli esterni
import numpy as np
import math
import cv2


def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    
    return np.dot(camera_parameters, projection)


def to_vertices(vertices, index):
    return np.array([vertices[x - 1] for x in index])


# project cube or model
def render(img, obj, projection, model, scalingScale=6, color=False):
    defaultColor = (137, 27, 211)
    # maggiore il valore, maggiore lo scaling del modello
    vertices = obj.vertices
    # scalingMatrix si occupa dello scaling del modello.
    scalingMatrix = np.eye(3) * scalingScale
    h, w = model.shape

    # questo va funzionare il rendering dei vertici correttamente, somehow.
    cpos = np.dot(np.linalg.inv(projection[:, :3]), projection[:, 3])
    obj.faces.sort(key=lambda x: np.sqrt(np.sum((cpos - np.mean(np.array(to_vertices(vertices, x[0]))))**2)))

    for face in obj.faces:
        # in face ci sta una istanza del tipo (face, norms, texcoords, material)
        # e.g. [1718, 1710, 1720]
        face_vertices = face[0]
        """
        e.g. [[-4.0000000e-06 -6.8255070e+00  3.2242416e+01]
            [ 2.9868200e-01 -6.8255070e+00  3.2145370e+01]
            [ 1.5702500e-01 -6.9483030e+00  3.1950399e+01]]
        """
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        """
        e.g. [[-2.40000000e-05 -4.09530420e+01  1.93454496e+02]
            [ 1.79209200e+00 -4.09530420e+01  1.92872220e+02]
            [ 9.42150000e-01 -4.16898180e+01  1.91702394e+02]]
        """
        points = np.dot(points, scalingMatrix)
        # print ("points 1: " , points[0])
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        """
        e.g. [[174.999976 144.546958 193.454496]
            [176.792092 144.546958 192.87222 ]
            [175.94215  143.810182 191.702394]]
        """
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # print ("points 2: " , points[0])
        """
        e.g. points.reshape(-1,1,3): [[[174.999976 144.546958 193.454496]]

            [[176.792092 144.546958 192.87222 ]]

            [[175.94215  143.810182 191.702394]]]
        """
        """
        e.g. dts = [[[296.03188792   9.05657252]]

            [[298.87256546   9.87051668]]

            [[297.73469973  11.22027995]]]
        """
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        """
        e.g. imgpts = [[[296   9]]

            [[298   9]]

            [[297  11]]]
        """
        imgpts = np.int32(dst)

        if color is False: 
            cv2.fillConvexPoly(img, imgpts, defaultColor)
        else: 
            tmpObj = obj.mtl[face[-1]]
            color = [x * 255 for x in tmpObj["Kd"]]
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img