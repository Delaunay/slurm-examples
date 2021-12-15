import numpy
import torch.nn as nn


class LeNet(nn.Module):
    """
    `Paper <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.
    Attributes
    ----------
    input_size: (1, 28, 28), (3, 32, 32), (3, 64, 64)
        Supported input sizes
    References
    ----------
    .. [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
        "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE, 86(11):2278-2324, November 1998.
    """

    def __init__(self, input_size, num_classes):
        super(LeNet, self).__init__()

        if not isinstance(num_classes, int):
            num_classes = numpy.product(num_classes)

        size = tuple(input_size)[1:]
        n_channels = input_size[0]

        if size == (28, 28):  # MNIST
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(50 * 4 * 4, 500)
            self.fc2 = nn.Linear(500, num_classes)

        elif size == (32, 32):  # CIFAR 10
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(50 * 5 * 5, 500)
            self.fc2 = nn.Linear(500, num_classes)

        elif size == (64, 64):  # TinyImageNet
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(3, 3)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(3, 3)
            self.fc1 = nn.Linear(50 * 5 * 5, 500)
            self.fc2 = nn.Linear(500, num_classes)
        else:
            raise ValueError(
                "There is no LeNet architecture for an input size {}".format(input_size)
            )

    def forward(self, batch):
        """Foward pass of the model"""
        out = nn.functional.relu(self.conv1(batch))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out
