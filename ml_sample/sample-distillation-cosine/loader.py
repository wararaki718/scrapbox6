from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


class DataLoaderFactory:
    @classmethod
    def create(cls, is_train: bool, is_shuffle: bool, root: str="./data") -> DataLoader:
        transforms_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        dataset = datasets.CIFAR10(
            root=root,
            train=is_train,
            download=True,
            transform=transforms_cifar,
        )
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=is_shuffle,
            num_workers=2,
        )
        return loader
