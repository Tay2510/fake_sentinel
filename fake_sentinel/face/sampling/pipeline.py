import sys
import argparse

from fake_sentinel.data.query import load_dfdc_dataframe
from fake_sentinel.data.video import sample_video_frames
from fake_sentinel.face.detection.facenet.utils import load_raw_results
from fake_sentinel.face.sampling.algorithm import *
from fake_sentinel.utils import parallelize_dataframe


def extract_face_images_from_video(video_file_path, data_dir, downsample=False):

    sample_id = Path(video_file_path).stem
    sample_dir = Path(data_dir) / sample_id

    if not sample_dir.is_dir():
        sample_dir.mkdir()

    facenet_results = load_raw_results(sample_id)
    frames = sample_video_frames(video_file_path, facenet_results.indices)

    tracked_faces, tracked_frame_indices = get_tracked_faces(frames, facenet_results)

    if len(tracked_faces) > 0:
        df_scores = score_faces(tracked_faces, frames.shape)

        sampled_tracked_faces = sample_tracked_faces(df_scores)

        for n, face_id in enumerate(sampled_tracked_faces):
            face_crops = crop_faces(frames, tracked_faces[face_id], tracked_frame_indices[face_id])

            stage_face_crops(face_crops, sample_dir / str(n), downsample=downsample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-p', '--processes', type=int, required=False, default=4)

    args = parser.parse_args(sys.argv[1:])

    if not Path(args.data_dir).is_dir():
        Path(args.data_dir).mkdir()

    finished_samples = [x.name for x in Path(args.data_dir).iterdir() if x.is_dir()]

    dataframe = load_dfdc_dataframe()
    dataframe = dataframe.loc[dataframe.index.difference(finished_samples)]

    dataframe_real = dataframe[dataframe['label'] == 'REAL']
    dataframe_fake = dataframe[dataframe['label'] == 'FAKE']

    def map_real(df):
        return df.filename.apply(lambda x: extract_face_images_from_video(x, args.data_dir, downsample=False))

    def map_fake(df):
        return df.filename.apply(lambda x: extract_face_images_from_video(x, args.data_dir, downsample=True))

    print('Staging REAL face crops from {:,} videos ...\n'.format(len(dataframe_real)))
    parallelize_dataframe(dataframe_real, map_real, n_cores=args.processes)

    print('Staging FAKE face crops from {:,} videos (with down-sampling) ...\n'.format(len(dataframe_fake)))
    parallelize_dataframe(dataframe_fake, map_fake, n_cores=args.processes)
