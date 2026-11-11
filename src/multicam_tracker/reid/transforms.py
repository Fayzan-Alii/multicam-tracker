from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def train_transform(size, pad: int = 10, erasing: float = 0.5):
    size = tuple(size)
    return T.Compose([
        T.Resize(size),
        T.RandomHorizontalFlip(0.5),
        T.Pad(pad),
        T.RandomCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=erasing, value=0),  # 0 after normalisation = the ImageNet mean
    ])


def test_transform(size):
    return T.Compose([T.Resize(tuple(size)), T.ToTensor(), T.Normalize(MEAN, STD)])
