import io
import base64
import cv2
import numpy as np
from pathlib import Path
from IPython.display import HTML


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


def display_in_jupyter(video_path, width=880, height=495):
    if video_path:
        video = io.open(str(video_path), 'r+b').read()
        encoded = base64.b64encode(video)

        handle = HTML(data="""
        <video  width={w} height={h} controls>
            <source src="data:video/mp4;base64,{stream}" type="video/mp4" />
        </video>""".format(w=width, h=height, stream=encoded.decode('ascii')))

        return handle
