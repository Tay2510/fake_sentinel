from torchvision import transforms
from fake_sentinel.pipeline.configs import CONFIGS, INPUT_SHAPES

INPUT_SHAPE = INPUT_SHAPES[CONFIGS['MODEL_NAME']]


def get_image_transforms(mode='train', input_shape=INPUT_SHAPE):
    preprocessoing = [transforms.ToPILImage(), transforms.Resize(input_shape)]
    normalization = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if mode == 'train':
        augmentation = CONFIGS['AUGMENTATION']
    else:
        augmentation = []

    transform_sequence = preprocessoing + augmentation + normalization

    return transforms.Compose(transform_sequence)
