import cv2
import numpy as np
from pathlib import Path


def video_to_frames(video_path, sampling_interval=1):
    if not Path(video_path).is_file():
        raise FileNotFoundError

    vidcap = cv2.VideoCapture(str(video_path))

    frames = []
    frame_count = 0

    while vidcap.grab():
        frame_count += 1

        if frame_count % sampling_interval == 0:
            _, frame = vidcap.retrieve()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    vidcap.release()

    return np.array(frames)
