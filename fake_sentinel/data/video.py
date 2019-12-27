import cv2
import numpy as np
from pathlib import Path


def video_to_frames(video_path):
    if not Path(video_path).is_file():
        raise FileNotFoundError

    vidcap = cv2.VideoCapture(str(video_path))

    frames = []

    while vidcap.grab():
        _, frame = vidcap.retrieve()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    vidcap.release()

    return np.array(frames)
