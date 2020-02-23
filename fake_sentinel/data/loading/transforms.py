from torchvision import transforms


IMAGE_TRANSFORMS = {
    'train': transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    ),

    'val': transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    ),
}
