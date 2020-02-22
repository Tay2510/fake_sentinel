import pickle
import random
from pathlib import Path

from fake_sentinel.data.paths import FACE_CROP_DIR, FACE_CROP_TABLE


class CropSampler:
    def __init__(self):
        with Path(FACE_CROP_TABLE).open('rb') as f:
            self.crop_table = pickle.load(f)

    def sample_from(self, sample_id):
        face_crops = self.crop_table[sample_id]
        face_ids = list(face_crops.keys())

        if len(face_ids) > 1:
            face_id = random.choice(face_ids)
        else:
            face_id = face_ids[0]

        return Path(FACE_CROP_DIR) / sample_id / str(face_id) / random.choice(face_crops[face_id])
