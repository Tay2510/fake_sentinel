import torch
import numpy as np
from facenet_pytorch import MTCNN

from fake_sentinel.data.video import video_to_frames


def chunks(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


class FaceExtractor:
    def __init__(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Running on device: {device}'.format(device=device))
        self.mtcnn = MTCNN(device=device).eval()
        self.gpu_batch_limit = 30

    def process(self, video_path, sampling_interval=1):
        frames = video_to_frames(video_path, sampling_interval)
        B = np.array([])
        P = np.array([])
        for chunk in chunks(frames, self.gpu_batch_limit):
            boxes, probs = self.mtcnn.detect(chunk)
            B = np.append(B, boxes)
            P = np.append(P, probs)

        return B, P
