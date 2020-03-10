import pickle
import random
import torch
from pathlib import Path
from torch.utils.data.sampler import Sampler

from fake_sentinel.data.paths import FACE_CROP_DIR, FACE_CROP_TABLE


class CropSampler:
    def __init__(self, mode='train'):
        self.mode = mode

        with Path(FACE_CROP_TABLE).open('rb') as f:
            self.crop_table = pickle.load(f)

    def sample_from(self, sample_id):
        face_crops = self.crop_table[sample_id]
        face_ids = list(face_crops.keys())

        if len(face_ids) > 1 and self.mode == 'train':
            face_id = random.choice(face_ids)
        else:
            face_id = face_ids[0]

        if self.mode == 'train':
            crop_path = random.choice(face_crops[face_id])
        else:
            crop_path = face_crops[face_id][0]

        return Path(FACE_CROP_DIR) / sample_id / str(face_id) / crop_path


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.real_indices = self.dataset.real_indices
        self.length = len(self.real_indices)
        self.shuffle = shuffle

    def __iter__(self):
        batch = []
        if self.shuffle:
            iter_list = [self.real_indices[i] for i in torch.randperm(len(self.real_indices)).tolist()]
        else:
            iter_list = self.real_indices

        for idx in iter_list:
            fake_idx = self.dataset.sample_fake(idx)
            batch.append(idx)
            batch.append(fake_idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        effective_batch_size = self.batch_size // 2

        if self.drop_last:
            return self.length // effective_batch_size
        else:
            return (self.length + effective_batch_size - 1) // effective_batch_size
