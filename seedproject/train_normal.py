import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from seedproject.models.lenet import LeNet

log = logging.getLogger()


def option(path, default=None, vtype=str):
    """Fetch a configurable value in the environment"""

    path = path.replace(".", "_").upper()
    full = f"SEEDPROJECT_{path}"
    value = vtype(os.environ.get(full, default))
    log.info("Using %s=%s", full, value)
    return value


class DistributedProcessGroup:
    """Helper to manage distributed training setup"""

    INSTANCE = None

    def __init__(self, backend="gloo"):
        self.__rank = int(os.environ.get("LOCAL_RANK", -1))

        if self.__rank >= 0:
            log.info("Initializing process group")
            dist.init_process_group(backend)
            log.info("Process group initialized")

        assert DistributedProcessGroup.INSTANCE is None
        DistributedProcessGroup.INSTANCE = self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.shutdown()

    @property
    def rank(self):
        """Return the current rank of our script -1 if running as a single GPU"""
        return self.__rank

    def shutdown(self):
        """Close the process group if running in distributed mode"""
        if self.__rank >= 0:
            log.info("Process group shutdown")
            dist.destroy_process_group()

    def device_id(self):
        """Return the device id this script should use"""
        if self.rank < 0:
            return -1

        return self.rank % torch.cuda.device_count()

    def device(self):
        """Return the device this scrupt should use"""
        if self.rank < 0:
            return torch.device("cuda")

        return torch.device(f"cuda:{self.device_id()}")


def rank():
    """Returns current rank"""
    group = DistributedProcessGroup.INSTANCE
    if group is None:
        return -1
    return group.rank


def device_id():
    """Returns current device_id"""
    group = DistributedProcessGroup.INSTANCE
    if group is None:
        return -1
    return group.device_id()


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

        if rank() >= 0:
            msg.append(f"Rank {rank()}")

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


class Checkpoint:
    """Basic checkpointer that saves all the object it is given periodically"""

    def __init__(self, path, name, every=2, **kwargs):
        self.data = kwargs
        self.every = every
        self.path = path
        self.name = name

    def end_epoch(self, epoch):
        """Called when the epoch finishes.
        Used to determined if we should save a new checkpoint or not

        """
        if epoch % self.every > 0:
            return
        self.save_checkpoint()

    def load_checkpoint(self):
        """Load a save state to resume training"""
        map_location = None

        path = os.path.join(self.path, self.name + ".chkpt")
        if not os.path.exists(path):
            log.info("No checkpoint found")
            self.save_checkpoint()
            return

        if rank() >= 0:
            # wait for rank 0 to save the checkpoint
            dist.barrier()

        # the other workers need to load the checkpoint
        if rank() > 0:
            # this is made to make it work with a single GPU
            # with 2 processes on a single GPU
            # for testing purposes
            map_location = {"cuda:%d" % 0: "cuda:%d" % device_id()}

        log.info("Loading checkpoint")
        state_dict = torch.load(
            path,
            map_location=map_location,
        )

        for key, value in self.data.items():
            if key not in state_dict:
                continue

            state = state_dict[key]

            if hasattr(value, "load_state_dict"):
                value.load_state_dict(state)
            else:
                state_dict[key] = state

    def save_checkpoint(self):
        """Save the current state of the trained to make it resumable"""
        log.info("save checkpoint")

        # only rank 0 can save the model
        if rank() > 0:
            return

        state_dict = dict()

        for key, value in self.data.items():
            if hasattr(value, "state_dict"):
                state_dict[key] = value.state_dict()
            else:
                state_dict[key] = value

        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, self.name + ".chkpt")

        # Save to a temporary file and then move it into the
        # final file, this is to prevent writing bad checkpoint
        # as move is atomic
        # in case of a failure the last good checkpoint is not going
        # to be corrupted
        _, name = tempfile.mkstemp(dir=os.path.dirname(path))

        torch.save(state_dict, name)

        # name and path need to be on the same filesystem on POSIX
        # (mv requirement)
        os.replace(name, path)


def dataparallel(model, device=None):
    """Wrap the model to make it parallel if rank is not none"""
    if rank() >= 0:
        log.info("enabling multi-gpu")
        return DistributedDataParallel(model, device_ids=[device_id()])

    return model


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

        self.trainset = CIFAR10(
            option("dataset.path", "/tmp/datasets"),
            train=True,
            download=True,
            transform=self.train_transform,
        )
        self.testset = CIFAR10(
            option("dataset.path", "/tmp/datasets"),
            train=False,
            download=True,
            transform=self.test_transform,
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

        self.testloader = DataLoader(
            self.testset,
            batch_size=2048,
            num_workers=n_worker,
        )

        self.local = LeNet(
            input_size=(3, 32, 32),
            num_classes=10,
        ).to(self.device)
        self.classifier = dataparallel(self.local, self.device)

        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        self.checkpoint = Checkpoint(
            path=option("checkpoint.path", "/tmp/chkpt"),
            name="LeNetModel",
            every=10,  # Every 10 epochs
            # Object to save in our checkpoint
            model=self.local,  # Save the module BEFORE the DataParallel wrapper
            optimizer=self.optimizer,
            stats=self.stats,
        )

    def start_epoch(self, epoch):
        """Called right before the start of an epoch"""
        if rank() > 0:
            return

        self.stats.start_epoch()

    def end_epoch(self, epoch):
        """Called right after the end of an epoch"""
        if rank() > 0:
            return

        self.stats.compute_test(
            self.testloader, self.classifier, self.criterion, self.device
        )
        self.stats.end_epoch()
        self.checkpoint.end_epoch(epoch)

    def start_step(self, step):
        """Called before a model process the next batch"""
        if rank() > 0:
            return

    def end_step(self, step):
        """Called after the model weights were update after a step"""
        if rank() > 0:
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


def fetch_device():
    """Set the default device to CPU if cuda is not available"""
    default = "cpu"
    if torch.cuda.is_available():
        default = "cuda"

    if rank() >= 0:
        torch.device(f"{default}:{device_id()}")

    return torch.device(default)


def main():
    """Run the trainer until completion"""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs")
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of dataloader worker"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight Decay"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Force cuda, fails if not present",
    )
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    setup_logging(args.verbose)

    with DistributedProcessGroup():
        task = Classification(
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            device=fetch_device(),
            n_worker=args.workers,
            batch_size=args.batch_size,
        )

        task.train(args.epochs)
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
