import random
from torch.utils.data import Dataset

from fake_sentinel.data.utils.image_utils import read_image
from fake_sentinel.data.loading.sampler import CropSampler
from fake_sentinel.data.loading.transforms import INCEPTION_TRANSFORMS

LABEL_ENCODER = {
    'FAKE': 1,
    'REAL': 0
}

SMOOTHING_EPISLON = 0.05


class FaceCropDataset(Dataset):
    def __init__(self, dataframe, mode='train'):
        dataframe = dataframe.reset_index(drop=True)
        self.sampler = CropSampler()
        self.image_transforms = INCEPTION_TRANSFORMS[mode]
        self.ids = dataframe['index'].to_list()
        self.labels = dataframe['label'].to_list()
        self.fake_mapping = build_real_fake_index_mapping(dataframe)
        self.real_indices = list(self.fake_mapping.keys())
        self.length = len(dataframe)

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
        y = label * (1 - SMOOTHING_EPISLON) + (1 - label) * SMOOTHING_EPISLON

        return X, y


def build_real_fake_index_mapping(df):
    real_idx_map = df[df['label'] == 'REAL']['index'].to_dict()
    real_idx_map = {v: k for k, v in real_idx_map.items()}

    original_groups = df[df['label'] == 'FAKE'].groupby('original', sort=False)

    mapping = {real_idx_map[k]: list(v) for k, v in original_groups.groups.items()}

    return mapping
