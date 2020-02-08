import pickle
from pathlib import Path
from collections import namedtuple

RESULT_DIR = Path('/home/jeremy/data/kaggle/dfdc_faces')

FaceNetResult = namedtuple('FaceNetResult', ['indices', 'boxes', 'probs', 'landmarks'])


def load_raw_results(sample_id):
    sample_dir = RESULT_DIR / sample_id

    boxes = pickle.load(Path(sample_dir / 'boxes.pkl').open('rb'))
    indices = pickle.load(Path(sample_dir / 'indices.pkl').open('rb'))
    landmarks = pickle.load(Path(sample_dir / 'landmarks.pkl').open('rb'))
    probs = pickle.load(Path(sample_dir / 'probs.pkl').open('rb'))

    boxes = merge_batch(boxes)
    landmarks = merge_batch(landmarks)
    probs = merge_batch(probs)

    assert len(indices) == len(boxes) == len(probs) == len(landmarks)

    return FaceNetResult(indices, boxes, probs, landmarks)


def merge_batch(batch_result):
    result = []
    for batch in batch_result:
        for data in batch:
            result.append(data)
    return result

