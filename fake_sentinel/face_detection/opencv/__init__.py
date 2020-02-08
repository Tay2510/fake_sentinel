import cv2
import numpy as np
from pathlib import Path

DIRECTORY = Path(__file__).parent

MODEL_FILE = DIRECTORY / 'deploy.prototxt.txt'
WEIGHATS_FILE = DIRECTORY / 'res10_300x300_ssd_iter_140000.caffemodel'

CHANNEL_MEAN = (104.0, 117.0, 123.0)
#CHANNEL_MEAN = (104.0, 177.0, 123.0)


class CVFaceDetector:
    def __init__(self, model_file=MODEL_FILE, weights_file=WEIGHATS_FILE, input_shape=(300, 300)):
        self.net = cv2.dnn.readNetFromCaffe(str(model_file), str(weights_file))
        self.input_shape = input_shape

    def detect(self, image, threshold=0.7):
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        blob = cv2.dnn.blobFromImage(cv2.resize(image, self.input_shape), 1.0, self.input_shape, CHANNEL_MEAN)
        self.net.setInput(blob)
        detections = self.net.forward()

        detections = detections[0][0][np.where(detections[0, 0, :, 2] > threshold)][:, 2:]
        detections[:, 1:] = detections[:, 1:] * np.array([w, h, w, h])

        return detections
