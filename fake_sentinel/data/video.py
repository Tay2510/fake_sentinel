import cv2
import numpy as np
from pathlib import Path


def video_to_frames(video_path):
    if not Path(video_path).is_file():
        raise FileNotFoundError

    vidcap = cv2.VideoCapture(str(video_path))

    success, image = vidcap.read()

    frames = []

    while success:
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        success, image = vidcap.read()

    vidcap.release()

    return np.array(frames)
