from torchvision import transforms
from fake_sentinel.pipeline.configs import CONFIGS


def get_image_transforms(mode='train'):
    transform_sequence = [
        transforms.ToPILImage(),
        transforms.Resize(CONFIGS['INPUT_SHAPE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if mode == 'train':
        transform_sequence.insert(2, transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_sequence)
