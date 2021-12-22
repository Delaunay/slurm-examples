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
            download = False
            src = os.path.join(CIFAR10.MILA_PATH)
            dst = os.path.join(root, CIFAR10.FOLDER)
            _copy(src, dst)

            for name in os.listdir(dst):
                print(name)

        train_dataset = datasets.CIFAR10(root=root, train=True, download=download)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=download)
        self.full = ConcatDataset([train_dataset, test_dataset])

        n = len(self.full)

        self.n_train = int(n * 0.60)
        self.n_test = len(test_dataset)
        self.n_valid = self.n_train - self.n_test

    def __getitem__(self, idx):
        return self.full[idx]

    def __len__(self):
        return len(self.full)

    def splits(self, final=False):
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
            trainset,
            validset,
            testset,
        )
