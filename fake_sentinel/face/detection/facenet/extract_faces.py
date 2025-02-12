import sys
import torch
import time
import argparse
from pathlib import Path
from facenet_pytorch import MTCNN

from fake_sentinel.data.query import load_dfdc_dataframe
from fake_sentinel.data.utils.video_utils import video_to_frames
from fake_sentinel.data.utils.data_utils import chunks, save_pickle


class FaceExtractor:
    def __init__(self, gpu_batch_limit=10):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Running on device: {device}'.format(device=device))
        self.mtcnn = MTCNN(device=device).eval()
        self.gpu_batch_limit = gpu_batch_limit

    def process(self, video_path, sampling_interval=1):
        frames, frame_indices = video_to_frames(video_path, sampling_interval, return_index=True)

        face_boxes = []
        face_probs = []
        face_landmarks = []

        for chunk in chunks(frames, self.gpu_batch_limit):
            boxes, probs, landmarks = self.mtcnn.detect(chunk, landmarks=True)
            face_boxes.append(boxes)
            face_probs.append(probs)
            face_landmarks.append(landmarks)

        return face_boxes, face_probs, face_landmarks, frame_indices


def run_pipeline(home_dir, sampling_interval, gpu_batch_limit=10):
    home_dir = Path(home_dir)
    df = load_dfdc_dataframe()
    face_extractor = FaceExtractor(gpu_batch_limit=gpu_batch_limit)

    if not home_dir.is_dir():
        home_dir.mkdir()

    total = len(df)

    for i, (sample_id, row) in enumerate(df.iterrows()):
        start = time.time()
        save_dir = home_dir / sample_id

        if save_dir.is_dir() and len(list(save_dir.glob('*.pkl'))) > 0:
            print('{:}/{:}'.format(i, total), sample_id, 'already processed')
        else:
            save_dir.mkdir(exist_ok=True)

            boxes, probs, landmarks, indices = face_extractor.process(row['filename'], sampling_interval=sampling_interval)

            save_pickle(boxes, save_dir / 'boxes.pkl')
            save_pickle(probs, save_dir / 'probs.pkl')
            save_pickle(landmarks, save_dir / 'landmarks.pkl')
            save_pickle(indices, save_dir / 'indices.pkl')

            elapse = time.time() - start
            print('{:}/{:}'.format(i, total), sample_id, '{} frames in {:.2f} sec'.format(len(indices), elapse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--save_dir')
    parser.add_argument('-i', '--interval', type=int, default=1)
    parser.add_argument('-g', '--gpu_batch_limit', type=int, default=10)

    args = parser.parse_args(sys.argv[1:])

    run_pipeline(args.save_dir, args.interval, args.gpu_batch_limit)
