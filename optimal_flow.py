import cv2
import numpy as np
import dlib


class face_landmarks:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.num_marks = 68
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def __call__(self, img):
        faces = self.detector(img)
        pos = []
        for face in faces:
            landmarks = self.predictor(image=img, box=face)
            for n in range(self.num_marks):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                pos.append([x, y])
        return pos


class dense_flow:

    def __init__(self):
        self.RLOF = cv2.optflow.DenseRLOFOpticalFlow_create()

    def __call__(self, prev, next):
        flow = self.RLOF.calc(prev, next, None)
        return np.transpose(flow, (2, 0, 1))
