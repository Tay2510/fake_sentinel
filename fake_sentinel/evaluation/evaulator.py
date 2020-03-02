import torch
import numpy as np
import random
from sklearn.metrics import log_loss

from fake_sentinel.face.detection.facenet.utils import merge_batch, FaceNetResult
from fake_sentinel.face.detection.facenet.extract_faces import FaceExtractor
from fake_sentinel.face.sampling.algorithm import get_tracked_faces, score_faces, sample_tracked_faces, crop_faces
from fake_sentinel.data.loading.transforms import INCEPTION_TRANSFORMS
from fake_sentinel.data.utils.video_utils import sample_video_frames
from fake_sentinel.data.query import load_dfdc_dataframe
from fake_sentinel.data.loading.dataset import LABEL_ENCODER
from fake_sentinel.model.classifier import create_classifier


def evaluate(model_path):
    df = load_dfdc_dataframe()
    df = df[df['split'] == 'val']
    targets = df['label'].apply(lambda x: LABEL_ENCODER[x])
    targets = np.array(targets, dtype=float)

    predicts = predict_videos(list(df['filename']), model_path)
    predicts = list(predicts.values())
    predicts = np.array(predicts, dtype=float)

    return log_loss(targets, predicts)


def predict_videos(filenames, model_path, sampling_interval=7, max_prediction_per_face=10):
    results = {}
    transform = INCEPTION_TRANSFORMS['val']
    classifier = create_classifier(pretrained=False)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

    detection_results = detect_faces(filenames, sampling_interval)

    for video_path, detections in zip(filenames, detection_results):
        frames = sample_video_frames(video_path, detections.indices)
        tracked_faces, tracked_frame_indices = get_tracked_faces(frames, detections)

        if len(tracked_faces) > 0:
            df_scores = score_faces(tracked_faces, frames.shape)

            sampled_tracked_faces = sample_tracked_faces(df_scores)

            confidence = 0

            for n, face_id in enumerate(sampled_tracked_faces):
                face_crops = crop_faces(frames, tracked_faces[face_id], tracked_frame_indices[face_id])
                face_crops = random.sample(face_crops, max_prediction_per_face)
                X = torch.stack([transform(f) for f in face_crops])
                logits = classifier(X)
                p = torch.nn.functional.softmax(logits, dim=-1).detach().numpy().mean(axis=0)
                confidence = max(confidence, p[1])

            results[video_path] = confidence

        else:
            results[video_path] = 1

    return results


def detect_faces(filenames, sampling_interval):
    face_extractor = FaceExtractor(gpu_batch_limit=10)
    results = []

    for video_path in filenames:
        face_boxes, face_probs, _, frame_indices = face_extractor.process(video_path, sampling_interval)
        face_boxes, face_probs = merge_batch(face_boxes), merge_batch(face_probs)

        results.append(FaceNetResult(frame_indices, face_boxes, face_probs, None))

    return results
