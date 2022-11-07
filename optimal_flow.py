import cv2
import numpy as np
import dlib


def tensor2cv2(img):
    return np.transpose((img * 255 + 0.5).clip(0, 255).astype(np.uint8), (1, 2, 0))


def normalize(img):
    m, s = img.mean(), img.std()
    return (img - m) / s


class face_landmarks(object):

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.num_marks = 68
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def __call__(self, img):
        cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = self.detector(cv_img)
        pos = []
        for face in faces:
            landmarks = self.predictor(image=cv_img, box=face)
            for n in range(self.num_marks):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                pos.append([x, y])
        return np.array(pos)


class dense_flow(object):

    def __init__(self):
        self.RLOF = cv2.optflow.DenseRLOFOpticalFlow_create()
        self.TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

    def __call__(self, prev, next, opt="RLOF"):
        cv_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
        cv_next = cv2.cvtColor(next, cv2.COLOR_RGB2BGR)
        if opt == "RLOF":
            flow = self.RLOF.calc(cv_prev, cv_next, None)
        elif opt == "TVL1":
            cv_prev_gray = cv2.cvtColor(cv_prev , cv2.COLOR_BGR2GRAY)
            cv_next_gray = cv2.cvtColor(cv_next, cv2.COLOR_BGR2GRAY)
            flow = self.TVL1.calc(cv_prev_gray, cv_next_gray, None)
        else: ValueError()
        return np.transpose(flow, (2, 0, 1))
