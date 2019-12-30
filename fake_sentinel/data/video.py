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


def video_to_jupyter(video_path, width=704, height=396):
    if video_path:
        video = io.open(str(video_path), 'r+b').read()
        encoded = base64.b64encode(video)

        handle = HTML(data="""
        <video  width={w} height={h} controls muted>
            <source src="data:video/mp4;base64,{stream}" type="video/mp4" />
        </video>""".format(w=width, h=height, stream=encoded.decode('ascii')))

        return handle


def join_videos(video_path_1, video_path_2, result_path):
    video_1 = video_to_frames(video_path_1)
    video_2 = video_to_frames(video_path_2)
    result = np.concatenate((video_1, video_2), axis=2)

    frames_to_video(result, result_path)


def frames_to_video(frames, save_path, fps=30):
    import imageio

    writer = imageio.get_writer(save_path, fps=fps, macro_block_size=1)

    for f in frames:
        writer.append_data(f)

    writer.close()
