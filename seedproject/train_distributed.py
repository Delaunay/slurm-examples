import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.multiprocessing.errors import record

import numpy as np


rank = int(os.environ.get("LOCAL_RANK", 0))
group = int(os.environ.get("GROUP_RANK", 0))
grank = int(os.environ.get("RANK", 0))

PATH = "./cifar_net.pth"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def option(path, default=None, vtype=str):
    """Fetch a configurable value in the environment"""

    path = path.replace(".", "_").upper()
    full = f"SEEDPROJECT_{path}"
    value = vtype(os.environ.get(full, default))
    return value


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@record
def train(args):
    if rank == 0:
        torchvision.datasets.CIFAR10(
            root=option("dataset.dest", "/tmp/datasets/cifar10"),
            train=True,
            download=rank == 0,
            transform=transform,
        )

    # Wait for the main worker to finish downloading the dataset
    dist.barrier()

    trainset = torchvision.datasets.CIFAR10(
        root=option("dataset.dest", "/tmp/datasets/cifar10"),
        train=True,
        download=False,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2048 * torch.cuda.device_count(),
        shuffle=True,
        num_workers=2,
    )

    device = torch.device(f"cuda:{rank}")
    model = Net().to(device=device)

    if grank <= 0:
        torch.save(model.state_dict(), PATH)

    dist.barrier()

    if grank > 0:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}

        weights = torch.load(PATH, map_location=map_location)
        model.load_state_dict(weights)

    dist.barrier()

    net = DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    losses = []

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

            losses.append(loss.detach())

        loss = sum([l.item() for l in losses]) / len(losses)

    print("Finished Training")

    try:
        from orion.client import report_objective

        report_objective(loss, name="loss")
    except ImportError:
        print("Orion is not installed")

    # Only one worker should save the network
    if grank <= 0:
        torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    # Usage:
    #
    #   # Works with a single GPU
    #   python distributed.py
    #
    #   # Works with multiple GPUs
    #   torchrun                            \
    #       --nproc_per_node=$GPU_COUNT     \
    #       --nnodes=$WORLD_SIZE            \
    #       --rdzv_id=$SLURM_JOB_ID         \
    #       --rdzv_backend=c10d             \
    #       --rdzv_endpoint=$RDV_ADDR       \
    #       distributed.py
    #
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--config", type=str, default=None, help="")
    args = parser.parse_args()

    if args.config is not None:
        import json

        with open(args.config, "r") as fp:
            config = json.load(fp)

        args = vars(args)
        args.update(config)
        args.pop("config")
        args = argparse.Namespace(**args)
        print(args)

    dist.init_process_group("gloo")

    train(args)

    dist.destroy_process_group()
