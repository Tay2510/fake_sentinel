import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import defaultdict

from fake_sentinel.face.tracking.matcher import Sort
from .meta import *


def score_faces(tracked_faces, frame_shape):
    tracked_face_scores = {}
    tracked_face_sizes = {}
    tracked_face_detections = {}

    for face_id, tracked_face_boxes in tracked_faces.items():
        detection_rate = len(tracked_face_boxes) / frame_shape[0]

        average_face_size = calculate_average_object_ratio(tracked_face_boxes, frame_shape[1:3])

        score = scoring_function(detection_rate, average_face_size)

        tracked_face_scores[face_id] = score
        tracked_face_sizes[face_id] = average_face_size
        tracked_face_detections[face_id] = detection_rate

    df_scores = pd.DataFrame({'score': pd.Series(tracked_face_scores),
                              'size': pd.Series(tracked_face_sizes),
                              'detection': pd.Series(tracked_face_detections)})

    return df_scores


def sample_tracked_faces(df_scores):
    n_sampled_faces = 1

    df_scores_sorted = df_scores.sort_values('score', ascending=False)

    best_face = df_scores_sorted.iloc[0].name

    if len(df_scores) > 1:
        second_best_face = df_scores_sorted.iloc[1].name

        size_is_close = np.absolute(df_scores.loc[best_face]['size'] - df_scores.loc[second_best_face]['size']) < MAX_FACE_SIZE_DIFF

        detection_is_okay = df_scores.loc[second_best_face]['detection'] > MIN_DET_RATE

        if size_is_close and detection_is_okay:
            n_sampled_faces = 2

    return df_scores_sorted.iloc[:n_sampled_faces].index.to_list()


def scoring_function(detection_rate, average_size):

    score = detection_rate / (np.exp(2 * np.abs((average_size - SIZE_AVG) / SIZE_AVG)))

    return score


def crop_faces(frames, tracked_face, tracked_frame_index):
    face_crops = []

    offset_fraction = (1 + CROP_ENLARGE_RATIO) / 2

    for i, b in zip(tracked_frame_index, tracked_face):
        w , h = b[2] - b[0], b[3] - b[1]
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

        new_b = [c[0] - offset_fraction * w,
                 c[1] - offset_fraction * h,
                 c[0] + offset_fraction * w,
                 c[1] + offset_fraction * h]

        frame = frames[i]

        f_h, f_w, _ = frame.shape

        new_b = [max(new_b[0], 0), max(new_b[1], 0), min(new_b[2], f_w), min(new_b[3], f_h)]

        b = np.array(new_b, dtype=int)

        crop = frame[b[1]:b[3], b[0]:b[2], :]

        face_crops.append(crop)

    return face_crops


def stage_face_crops(face_crops, crop_dir):
    crop_dir = Path(crop_dir)

    if not crop_dir.is_dir():
        crop_dir.mkdir()

    for n, crop in enumerate(face_crops):
        save_path = crop_dir / '{}.png'.format(n)
        img = Image.fromarray(crop)
        img.save(str(save_path))


def calculate_average_object_ratio(boxes, frame_shape):
    obj_array = np.array(boxes)
    obj_w = obj_array[:, 2] - obj_array[:, 0]
    obj_h = obj_array[:, 3] - obj_array[:, 1]
    h, w = frame_shape

    return np.sqrt(obj_w.mean() / w * obj_h.mean() / h)


def get_tracked_faces(frames, facenet_results):
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THR)

    all_faces = defaultdict(list)
    frame_indices = defaultdict(list)

    for i in range(len(frames)):
        boxes = facenet_results.boxes[i]
        probs = facenet_results.probs[i]

        if boxes is not None:
            dets = np.concatenate([boxes, probs.reshape(len(probs), 1)], axis=1)
            faces = tracker.update(dets)
            for f in faces:
                all_faces[int(f[-1])].append(f[:4])
                frame_indices[int(f[-1])].append(i)

    return dict(all_faces), dict(frame_indices)
