import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from orion.client import report_objective

import seedproject.distributed.distributed as dist
from seedproject.models.lenet import LeNet
from seedproject.dataset.CIFAR10 import CIFAR10
from seedproject.checkpoint import Checkpoint

log = logging.getLogger()


def option(path, default=None, vtype=str):
    """Fetch a configurable value in the environment"""

    path = path.replace(".", "_").upper()
    full = f"SEEDPROJECT_{path}"
    value = vtype(os.environ.get(full, default))
    log.info("Using %s=%s", full, value)
    return value


@dataclass
class Stats:
    """You can use this to save the data you need offline and send it to your experiment
    management backend

    """

    epoch_start: int = 0
    epoch_time: int = 0
    online_losses: list = field(default_factory=list)
    online_loss: float = None
    test_loss: float = None
    test_accuracy: float = None
    series: dict = field(default_factory=lambda: defaultdict(list))
    metrics: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)

    def add(self, name, key, value):
        """Insert a time serie value

        Examples
        --------

        >>> s = Stats()
        >>> s.add('accuracy', 0, 0.10)
        >>> s.add('accuracy', 10, 0.90)

        """
        self.series[name].append((key, value))

    def add_tags(self, *args):
        """Add tags

        Examples
        --------

        >>> s = Stats()
        >>> s.add_tags('LeNet', 'classification')

        """
        self.tags.extend(args)

    def value(self, **kwargs):
        """Add values we want to save for later"""
        self.metrics.update(kwargs)

    def report(self):
        """Generate a progress report"""
        msg = []

        if dist.rank() >= 0:
            msg.append(f"Rank {dist.rank()}")

        if self.epoch_time:
            msg.append(f"Epoch Time {self.epoch_time:.2f}")

        if self.test_accuracy:
            msg.append(f"Acc: {self.test_accuracy * 100:6.2f}")

        if self.test_loss:
            msg.append(f" Loss: {self.test_loss:.4f} ")

        return " ".join(msg)

    def summary(self):
        """Print a summary of all the metrics we kept track of"""
        values = []
        for key, value in self.metrics.items():
            values.append(f"{key:>30}: {value}")

        for key, value in self.series.items():
            key = f"series.{value}"
            values.append(f"{key:>30}: {value[-1]}")

        return "\n".join(values)

    def show(self):
        """Show a progress report"""
        print("\r" + self.report(), end="")

    def start_epoch(self):
        """Called at the start of an epoch"""
        self.epoch_start = time.time()

    def end_epoch(self):
        """Called at the end of an epoch"""
        self.update_loss()
        self.epoch_time = time.time() - self.epoch_start
        self.show()

    def add_step_loss(self, loss):
        """Keep track of our online losses,

        We use detach so the compute graph diff graph is not impacted by it.
        We do not use ``.item()`` to prevent a cuda sync.
        We will wait until the end of the epoch to accumulate all the partial losses.
        """
        self.online_losses.append(loss.detach())

    def update_loss(self):
        """Make a summation of all the loss we gathered so far"""
        self.online_loss = sum([loss.item() for loss in self.online_losses]) / len(
            self.online_losses
        )
        self.online_losses = []

    def compute_test(self, testset, model, criterion, device):
        """Compute the test accuracy on a frozen/eval/inference model"""

        model.eval()
        with torch.no_grad():
            values = []
            count = 0
            acc = 0

            for obs, target in testset:
                obs, target = obs.to(device), target.to(device)

                scores = model(obs)
                loss = criterion(scores, target)
                values.append(loss.detach())

                _, predicted = torch.max(scores, 1)
                acc += (predicted == target).sum()
                count += len(target)

            self.test_loss = sum([loss.item() for loss in values]) / len(values)
            self.test_accuracy = acc / count
        model.train()


class Classification:
    """Train a given model to classify observation"""

    def __init__(
        self,
        lr=0.001,
        weight_decay=0.001,
        momentum=0.9,
        device=None,
        n_worker=4,
        batch_size=256,
        uid=None,
    ):
        self.device = device
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.train_transform = transforms.Compose(
            [
                # to_pil_image,
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if dist.has_dataset_autority():
            # download the dataset first or copy
            CIFAR10(option("dataset.dest", "/tmp/datasets/cifar10"), download=True)

        # wait for rank 0 to download the dataset
        dist.barrier()
        dataset = CIFAR10(option("dataset.dest", "/tmp/datasets/cifar10"))
        self.trainset, self.validset, self.testset = dataset.splits(
            train_transform=self.train_transform,
            test_transform=self.test_transform,
        )

        self.stats = Stats()

        self.criterion = CrossEntropyLoss()
        self.sampler = RandomSampler(self.trainset)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=n_worker,
            sampler=self.sampler,
        )

        self.validloader = DataLoader(
            self.validset,
            batch_size=2048,
            num_workers=n_worker,
        )

        self.local = LeNet(
            input_size=(3, 32, 32),
            num_classes=10,
        ).to(self.device)
        self.classifier = dist.dataparallel(self.local, self.device)

        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        self.checkpoint = Checkpoint(
            path=option("checkpoint.path", "/tmp/chkpt"),
            name=f"LeNetModel_{uid}",
            every=10,  # Every 10 epochs
            # Object to save in our checkpoint
            model=self.local,  # Save the module BEFORE the DataParallel wrapper
            optimizer=self.optimizer,
            stats=self.stats,
        )

    def start_epoch(self, epoch):
        """Called right before the start of an epoch"""
        if not dist.has_weight_autority():
            return

        self.stats.start_epoch()

    def end_epoch(self, epoch):
        """Called right after the end of an epoch"""
        if not dist.has_weight_autority():
            return

        self.stats.compute_test(
            self.validloader, self.classifier, self.criterion, self.device
        )
        self.stats.end_epoch()
        self.checkpoint.end_epoch(epoch)

    def start_step(self, step):
        """Called before a model process the next batch"""
        if not dist.has_weight_autority():
            return

    def end_step(self, step):
        """Called after the model weights were update after a step"""
        if not dist.has_weight_autority():
            return

    def start_train(self):
        """Called when train starts"""
        self.checkpoint.load_checkpoint()

    def end_train(self):
        """Called when training has finished"""
        self.checkpoint.save_checkpoint()
        print()

    def train(self, epochs):
        """Train for epochs"""
        self.start_train()
        start = 0

        for epoch in range(start, epochs):
            self.epoch(epoch)

        self.end_train()

    def epoch(self, epoch):
        """Do a full epoch once"""
        self.start_epoch(epoch)

        for step, mini_batch in enumerate(self.trainloader):
            self.start_step(step)

            self.step(step, mini_batch)

            self.end_step(step)

        self.end_epoch(epoch)

    def step(self, step, batch):
        """Do a single optimization step"""
        self.classifier.train()
        self.optimizer.zero_grad()

        batch, target = batch
        predictions = self.classifier(batch.to(self.device))
        loss = self.criterion(predictions, target.to(device=self.device))
        loss.backward()

        self.optimizer.step()
        self.stats.add_step_loss(loss.detach())


def setup_logging(verbose):
    """Configure the logging level given a verbose count"""

    log_level = (5 - min(verbose, 4)) * 10
    level_name = logging.getLevelName(log_level)
    print(f"Setting log level: {level_name} ({log_level})")

    handler = logging.StreamHandler(sys.stderr)
    logging.basicConfig(level=level_name, handlers=[handler])


def compute_identity(sample, size):
    """Compute a unique hash out of a dictionary
    Parameters
    ----------
    sample: dict
        Dictionary to compute the hash from
    size: int
        size of the unique hash
    """
    import hashlib
    from collections import OrderedDict

    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode("utf8"))

        if isinstance(v, (dict, OrderedDict)):
            sample_hash.update(compute_identity(v, size).encode("utf8"))
        else:
            sample_hash.update(str(v).encode("utf8"))

    return sample_hash.hexdigest()[:size]


@dist.record
def main():
    """Run the trainer until completion"""
    from argparse import ArgumentParser, Namespace

    parser = ArgumentParser()
    parser.add_argument(
        "--batch-size", default=256, type=int, help="Batch size (per GPU)"
    )
    parser.add_argument("--epochs", default=10, type=int, help="Epochs")
    parser.add_argument(
        "--workers", default=2, type=int, help="Number of dataloader worker"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight Decay"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument(
        "--identity",
        default="manual",
        type=str,
        help="Unique name of the trial for checkpointing",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Force cuda, fails if not present",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="",
    )
    args = parser.parse_args()

    if args.config is not None:
        import json

        with open(args.config, "r") as fp:
            config = json.load(fp)

        identity = compute_identity(config, 16)

        args = vars(args)
        args.update(config)
        args.pop("config")
        args["identity"] = identity
        args = Namespace(**args)
        print(args)

    if args.cuda:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    setup_logging(args.verbose)

    with dist.DistributedProcessGroup():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.batch_size = args.batch_size * world_size

        print(f"  GPU batch: {args.batch_size // world_size}")
        print(f"Total batch: {args.batch_size}")

        task = Classification(
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            device=dist.device(),
            n_worker=args.workers,
            batch_size=args.batch_size,
            uid=args.identity,
        )

        task.train(args.epochs)

        report_objective(task.stats.test_loss, name="loss")
    #


if __name__ == "__main__":
    #
    #   MultiGPU
    #
    # You can test the script with a single GPU using the command below
    #
    #   torchrun --nproc_per_node=4 --nnodes=1 seedproject/train_normal.py -vvv
    #
    # if you have a single GPU:
    #   Launch 2 processes on the device:0
    #
    # if you have two GPU
    #   Launch 1 process for device:0
    #   Launch 1 process for device:1
    #

    #
    #   Single GPU
    #
    #   python seedproject/train_normal.py -vv
    #
    # Launch 2 processes on the device:0
    #
    main()
