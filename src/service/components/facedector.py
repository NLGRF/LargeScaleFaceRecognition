import  cv2
import  numpy as np
import  dlib
import time

class face_detector:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.tracker = cv2.TrackerMedianFlow_create()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.4
        self.fontColor = (255, 255, 255)
        self.lineType = 2
        self.onTrack = False
    


    def draw_bbox(self, cv_img):

        face_pos = []

        t0 = time.time()

        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray_img)

        fps = 1.0/ (time.time()-t0)

        fps_show = f"{fps}fps"

        if not self.onTrack:

            for face in faces:

                x = face.left()
                y = face.top()
                w = face.right() - x +10
                h = face.bottom() - y +10

                face_pos.append([x,y,w,h])

                print(face_pos)
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 25, 10), 2)

                if self.tracker.init(cv_img, (x, y, w, h)):
                    self.onTrack = True

        else:
            ok, bbox = self.tracker.update(cv_img)
            if ok:
         

                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.line(cv_img, (x, y), (x + int(w / 5), y), (0,255,00),3)
                cv2.line(cv_img, (x + int((w/ 5)* 4), y), (x + w, y), (0,255,0),3)
                cv2.line(cv_img, (x, y), (x, y + int(h / 5)), (0,255,10), 3)
                cv2.line(cv_img, (x +w, y), (x + w, y + int(h / 5)),(0,255,0), 3)
                cv2.line(cv_img, (x, (y + int(h /5* 4))), (x, y+h),(0,255,0), 3)
                cv2.line(cv_img, (x, (y+h)), (x + int(w / 5), y + h),(0,255,0), 3)
                cv2.line(cv_img, (x + int((w/ 5)* 4), y+ h), (x +w, y + h), (0,255,0), 3)
                cv2.line(cv_img, (x + w, (y + int(h / 5 * 4))), (x + w, y + h), (0, 255, 0), 3)
            else:
                self.onTrack = False
                self.tracker = cv2.TrackerMedianFlow_create()

        yield cv_img ,face_pos

