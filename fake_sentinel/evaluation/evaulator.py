import torch

from fake_sentinel.face.detection.facenet.utils import merge_batch, FaceNetResult
from fake_sentinel.face.detection.facenet.extract_faces import FaceExtractor
from fake_sentinel.data.utils.video_utils import sample_video_frames
from fake_sentinel.face.sampling.algorithm import get_tracked_faces, score_faces, sample_tracked_faces, crop_faces
from fake_sentinel.data.loading.transforms import INCEPTION_TRANSFORMS
from fake_sentinel.model.classifier import create_classifier


def predict_videos(filenames, model_path):
    results = {}
    transform = INCEPTION_TRANSFORMS['val']
    classifier = create_classifier(pretrained=False)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

    detection_results = detect_faces(filenames)

    for video_path, detections in detection_results.items():
        frames = sample_video_frames(video_path, detections.indices)
        tracked_faces, tracked_frame_indices = get_tracked_faces(frames, detections)

        if len(tracked_faces) > 0:
            df_scores = score_faces(tracked_faces, frames.shape)

            sampled_tracked_faces = sample_tracked_faces(df_scores)

            confidence = 0

            for n, face_id in enumerate(sampled_tracked_faces):
                face_crops = crop_faces(frames, tracked_faces[face_id], tracked_frame_indices[face_id])
                X = transform(face_crops[0])    # TODO: predict multiple face crops
                logits = classifier(X.unsqueeze(0))
                p = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()

                confidence = max(confidence, p.flatten()[1])

            results[video_path] = confidence
        else:
            results[video_path] = 1.0

    return results


def detect_faces(filenames):
    face_extractor = FaceExtractor(gpu_batch_limit=10)
    results = {}

    for video_path in filenames:
        face_boxes, face_probs, _, frame_indices = face_extractor.process(video_path, 3)
        face_boxes, face_probs = merge_batch(face_boxes), merge_batch(face_probs)

        results[video_path] = FaceNetResult(frame_indices, face_boxes, face_probs, None)

    return results
