import cv2
import numpy as np
import dlib


def tensor2cv2(img):
    return np.transpose((img * 255 + 0.5).clip(0, 255).astype(np.uint8), (1, 2, 0))


def normalize(img):
    m, s = img.mean(), img.std()
    return (img - m) / s


class FaceLandmarks(object):

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


class DenseFlow(object):

    def __init__(self):
        self.RLOF = cv2.optflow.DenseRLOFOpticalFlow_create()
        self.TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

    def __call__(self, prev, next, opt="RLOF"):
        cv_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
        cv_next = cv2.cvtColor(next, cv2.COLOR_RGB2BGR)
        cv_prev_gray = cv2.cvtColor(cv_prev, cv2.COLOR_BGR2GRAY)
        cv_next_gray = cv2.cvtColor(cv_next, cv2.COLOR_BGR2GRAY)
        if opt == "RLOF":
            try:
                flow = self.RLOF.calc(cv_prev, cv_next, None)
            except:
                flow = self.RLOF.calc(cv_prev_gray, cv_next_gray, None)
        elif opt == "TVL1":
            flow = self.TVL1.calc(cv_prev_gray, cv_next_gray, None)
        else:
            ValueError()
        return np.transpose(flow, (2, 0, 1))


def uniform_choose(iterable, num_sample):
    l = len(iterable)
    i, interval = 0, l / num_sample
    choice = []
    for _ in range(num_sample):
        choice.append(iterable[int(round(i))])
        i += interval
    return choice


class BatchFlow(object):

    def __init__(self, num_frames):
        self.denseflow = DenseFlow()
        self.landmarks = FaceLandmarks()
        self.num_frames = num_frames

    def __call__(self, imgs, seq_len):
        flows = []
        for (i, img) in enumerate(imgs):
            sample = uniform_choose(range(seq_len[i]), self.num_frames)
            frames = [img[j].clone().detach().cpu().numpy() for j in sample]
            flow = []
            for j in range(len(frames)):
                landmark = self.landmarks(frames[j])
                markmat = np.expand_dims(np.zeros_like(frames[j][:, :, 0]), 0)
                for x, y in landmark:
                    markmat[0][x][y] = 1
                flow.append(markmat)
                if j > 0:
                    try:
                        flow.append(self.denseflow(frames[j - 1], frames[j], "RLOF"))
                    except:
                        flow.append(self.denseflow(frames[j - 1], frames[j], "TVL1"))
            flow = np.concatenate(flow, axis=0)
            flows.append(flow)
        return np.array(flows)
