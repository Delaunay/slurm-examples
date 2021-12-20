import shutil
import os
import multiprocessing as mp


def _copy(src, dest):
    os.makedirs(dest, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


class CopyDataset:
    def __init__(self, dataset, src, dest=None, *args, **kwargs):
        # if src != dest and src is not None and dest is not None:
        #    _copy(src, dest)

        self.dataset = dataset(dest, *args, **kwargs)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
