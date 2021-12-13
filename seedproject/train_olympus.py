from olympus.datasets import DataLoader, Dataset, SplitDataset, register_dataset
from olympus.models import Model
from olympus.optimizers import Optimizer
from olympus.utils import fetch_device

from seedproject.models.lenet import LeNet

# Classification is also implemented by Olympus
# it is given here as an example
from seedproject.tasks.classification import Classification

register_dataset("MyModel", LeNet)


def main(validate=False):
    """Build the configuration we want to train on the cluster"""

    # -------------------------------------------------------------------------
    #   Prepare our dataset/model/optimizer
    # -------------------------------------------------------------------------
    # Split Dataset allow us to generate different split
    # than the official one
    dataset = SplitDataset(
        Dataset(
            "cifar10",
            path="/tmp/dataset",
        ),
        split_method="original",
    )

    # Dataloader will generate 3 dataloader for each split
    loader = DataLoader(
        dataset,
        sampler_seed=0,
        batch_size=256,
        valid_batch_size=2048,
        num_workers=4,
    )

    # Some models can accept different input size
    input_size, target_size = loader.get_shapes()
    model = Model(
        "MyModel",
        input_size=input_size,
        output_size=target_size[0],
    )

    optimizer = Optimizer("sgd")

    train, valid, test = loader.get_loaders(hpo_done=False)

    metrics = []
    append_if(metrics, validate and valid, Accuracy(name="validation", loader=valid))
    append_if(metrics, validate and test, Accuracy(name="test", loader=test))

    # -------------------------------------------------------------------------
    #   Prepare for training
    # -------------------------------------------------------------------------
    trainer = Classification(
        model,
        optimizer,
        train,
        device=fetch_device(),
        metrics=metrics,
    )

    # -------------------------------------------------------------------------
    #   Set our hyper-parameter
    # -------------------------------------------------------------------------

    trainer.init(
        optimizer=dict(
            weight_decay=0.001,
            lr=0.001,
            momentum=0.9,
        ),
        model=dict(),
    )

    # -------------------------------------------------------------------------
    #  Train
    # -------------------------------------------------------------------------

    trainer.fit(epochs=10)


if __name__ == "__main__":
    main()
