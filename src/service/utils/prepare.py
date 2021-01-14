import numpy as np
import cv2


def preprocess_image(image, face_pos, aspect_size=(160, 160)):

    crop_faces = []
    for face in face_pos:
        x = face[0],
        y = face[1],
        w = face[2],
        h = face[3],

        # print(x[0], y, w, h)
        # 
        crop_face = image[y[0] + 100 : h[0] + 100, x[0] + 100 : w[0] + 100]
        crop_face = np.expand_dims(crop_face, 0)

        crop_faces.append(crop_face)

    yield  crop_faces



    