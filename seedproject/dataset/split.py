from dataclasses import dataclass

import numpy


@dataclass
class Split:
    """Returns 3 Splits of the main data set. The splits are indices of the samples inside the data set"""

    train: numpy.array
    valid: numpy.array
    test: numpy.array

    def items(self):
        return ("train", self.train), ("valid", self.valid), ("test", self.test)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def keys(self):
        return "train", "valid", "test"

    def values(self):
        return self.train, self.valid, self.test


def split(datasets, data_size=None, seed=None, ratio=None, index=None, balanced=None):
    """Original dataset split"""
    n_train = datasets.train_size
    n_valid = datasets.valid_size
    n_test = datasets.test_size
    n_points = len(datasets)

    assert n_points == n_train + n_valid + n_test

    return Split(
        train=range(n_train),
        valid=range(n_train, n_train + n_valid),
        test=range(n_train + n_valid, n_points),
    )
