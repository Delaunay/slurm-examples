from torch.utils.data import ConcatDataset, Dataset, Subset
import torchvision.datasets as datasets

from seedproject.dataset.split import split
import shutil
import os


def _copy(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)


class CIFAR10(Dataset):
    FOLDER = "cifar10"
    MILA_PATH = "/network/datasets/cifar10.var/cifar10_torchvision/"

    def __init__(self, root, download=False):

        if os.path.exists(CIFAR10.MILA_PATH):
            # Copy our cached version locally

            # <root>/cifar10/cifar-10-batches-py/*data_batch_*
            #
            download = False
            src = os.path.join(CIFAR10.MILA_PATH)
            shutil.copytree(src, root, ignore=True, dirs_exist_ok=True)

        train_dataset = datasets.CIFAR10(root=root, train=True, download=download)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=download)
        self.full = ConcatDataset([train_dataset, test_dataset])

        n = len(self.full)

        self.train_size = int(n * 0.60)
        self.test_size = len(test_dataset)
        self.valid_size = len(train_dataset) - self.train_size

    def __getitem__(self, idx):
        return self.full[idx]

    def __len__(self):
        return len(self.full)

    def splits(self, final=False, train_transform=None, test_transform=None):
        splits = split(self)

        trainset = Subset(self, splits.train)
        validset = Subset(self, splits.valid)
        testset = Subset(self, splits.test)

        if final:
            trainset = ConcatDataset([trainset, validset])
            # Valid set get merged to get a bigger trainset
            # when HPO is done
            validset = None
        else:
            # Not allowed to use testset during HPO
            testset = None

        return (
            Transformed(trainset, train_transform) if trainset else None,
            Transformed(validset, test_transform) if validset else None,
            Transformed(testset, test_transform) if testset else None,
        )


class Transformed(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, idx):
        data, target = self.dataset[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.dataset)
