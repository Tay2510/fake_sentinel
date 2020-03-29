import random
import numpy as np
from torch.utils.data import Dataset

from fake_sentinel.data.utils.image_utils import read_image
from fake_sentinel.data.loading.sampler import CropSampler
from fake_sentinel.data.loading.transforms import get_image_transforms

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
            X = mix_lambda * X + (1 - mix_lambda) * another_X
            y = mix_lambda * y + (1 - mix_lambda) * another_y

        return X, y


def build_real_fake_index_mapping(df):
    real_idx_map = df[df['label'] == 'REAL']['index'].to_dict()
    real_idx_map = {v: k for k, v in real_idx_map.items()}

    original_groups = df[df['label'] == 'FAKE'].groupby('original', sort=False)

    mapping = {real_idx_map[k]: list(v) for k, v in original_groups.groups.items()}

    return mapping
