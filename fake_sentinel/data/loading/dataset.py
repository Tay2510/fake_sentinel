from torch.utils.data import Dataset

from fake_sentinel.data.utils.image_utils import read_image
from fake_sentinel.data.loading.sampler import CropSampler
from fake_sentinel.data.loading.transforms import IMAGE_TRANSFORMS

LABEL_ENCODER = {
    'FAKE': 1,
    'REAL': 0
}


class FaceCropDataset(Dataset):
    def __init__(self, dataframe, mode='train'):
        self.sampler = CropSampler(mode)
        self.image_transforms = IMAGE_TRANSFORMS[mode]
        self.ids = dataframe.index.to_list()
        self.labels = dataframe.label.to_list()
        self.length = len(dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample_id, sample_label = self.ids[idx], self.labels[idx]
        image_file = self.sampler.sample_from(sample_id)
        image = read_image(image_file)
        label = LABEL_ENCODER[sample_label]
        X = self.image_transforms(image)
        y = label

        return X, y
