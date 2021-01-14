import numpy as np
import cv2
from service.utils.prepare import preprocess_image
from service.components.facedector import face_detector

class main:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.face_detector = face_detector()

    def run(self):

        while True:
            ret, cv_img = self.cap.read()

            for cv_img, faces in self.face_detector.draw_bbox(cv_img):
                for face in preprocess_image(cv_img, faces):
                    print(face)
            
            # Display
            cv2.imshow('img', cv_img)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break

        cap.release()

if __name__ == '__main__':
    main().run()