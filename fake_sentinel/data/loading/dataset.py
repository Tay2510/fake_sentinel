import random
import numpy as np
from torch.utils.data import Dataset

from fake_sentinel.data.utils.image_utils import read_image
from fake_sentinel.data.loading.sampler import CropSampler
from fake_sentinel.data.loading.transforms import get_image_transforms, INPUT_SHAPE

LABEL_ENCODER = {
    'FAKE': 1,
    'REAL': 0
}


class FaceCropDataset(Dataset):
    def __init__(self, dataframe, mode='train', smoothing_epsilon=0, mixedup=-1):
        dataframe = dataframe.reset_index(drop=True)
        self.sampler = CropSampler()
        self.image_transforms = get_image_transforms(mode)
        self.ids = dataframe['index'].to_list()
        self.labels = dataframe['label'].to_list()
        self.fake_mapping = build_real_fake_index_mapping(dataframe)
        self.real_indices = list(self.fake_mapping.keys())
        self.length = len(dataframe)
        self.smoothing_epislon = smoothing_epsilon
        self.mixedup = mixedup

    def sample_fake(self, real_idx):
        candidates = self.fake_mapping[real_idx]
        return random.choice(candidates)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample_id, sample_label = self.ids[idx], self.labels[idx]
        image_file = self.sampler.sample_from(sample_id)
        image = read_image(image_file)
        label = LABEL_ENCODER[sample_label]
        X = self.image_transforms(image)
        y = label * (1 - self.smoothing_epislon) + (1 - label) * self.smoothing_epislon

        if self.mixedup > 0:
            mix_lambda = np.random.beta(self.mixedup, self.mixedup)
            another_idx = random.choice(range(self.length))
            another_id, another_label = self.ids[another_idx], self.labels[another_idx]
            another_image_file = self.sampler.sample_from(another_id)
            another_image = read_image(another_image_file)
            another_label = LABEL_ENCODER[another_label]
            another_X = self.image_transforms(another_image)
            another_y = another_label * (1 - self.smoothing_epislon) + (1 - another_label) * self.smoothing_epislon

            bbx1, bby1, bbx2, bby2 = rand_bbox(INPUT_SHAPE, mix_lambda)
            mix_lambda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (INPUT_SHAPE[0] * INPUT_SHAPE[1]))

            X[:, bbx1:bbx2, bby1:bby2] = another_X[:, bbx1:bbx2, bby1:bby2]
            y = mix_lambda * y + (1 - mix_lambda) * another_y

        return X, y


def build_real_fake_index_mapping(df):
    real_idx_map = df[df['label'] == 'REAL']['index'].to_dict()
    real_idx_map = {v: k for k, v in real_idx_map.items()}

    original_groups = df[df['label'] == 'FAKE'].groupby('original', sort=False)

    mapping = {real_idx_map[k]: list(v) for k, v in original_groups.groups.items()}

    return mapping


def rand_bbox(size, lam):
    W, H = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

