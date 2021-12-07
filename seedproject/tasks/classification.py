import torch
from torch.nn import Module, CrossEntropyLoss

from olympus.tasks.task import Task
from olympus.metrics import OnlineTrainAccuracy, Accuracy
from olympus.utils import select, drop_empty_key
from olympus.resuming import state_dict, load_state_dict
from olympus.observers import (
    ElapsedRealTime,
    SampleCount,
    ProgressView,
    Speed,
    CheckPointer,
)


class Classification(Task):
    def __init__(
        self, classifier, optimizer, dataloader, storage=None, metrics=None, device=None
    ):
        super(Classification, self).__init__(device=device)
        self.classifier = classifier
        self.optimizer = optimizer

        self.criterion = CrossEntropyLoss()
        self.dataloader = dataloader

        # Add the metrics we would like to track
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        self.metrics.append(OnlineTrainAccuracy())
        self.metrics.append(Speed())

        if metrics:
            for m in metrics:
                self.metrics.append(m)

        self.metrics.append(ProgressView(self.metrics.get("Speed")))

        # Add checkpointing if we have a valid storage location
        if storage:
            self.metrics.append(CheckPointer(storage=storage))

    def get_space(self):
        """Return hyper parameter space"""
        return drop_empty_key(
            {
                "optimizer": self.optimizer.get_space(),
                "model": self.classifier.get_space(),
            }
        )

    def get_current_space(self):
        """Get currently defined parameter space"""
        return {
            "optimizer": self.optimizer.get_current_space(),
            "model": self.classifier.get_current_space(),
        }

    def init(self, optimizer=None, model=None, uid=None):
        """Set our hyperparameters"""

        optimizer = select(optimizer, {})
        model = select(model, {})

        self.classifier.init(**model)

        # list of all parameters this task has
        parameters = self.classifier.parameters()

        # We need to set the device now so optimizer receive cuda tensors
        self.set_device(self.device)
        self.optimizer.init(params=parameters, override=True, **optimizer)

        self.hyper_parameters = {
            "optimizer": optimizer,
            "model": model,
        }

        # Get all hyper parameters even the one that were set manually
        hyperparameters = self.get_current_space()

        # Trial Creation and Trial resume
        self.metrics.new_trial(hyperparameters, uid)
        self.set_device(self.device)

    def fit(self, epochs, context=None):
        if self.stopped:
            return

        self.classifier.to(self.device)
        self._start(epochs)

        for epoch in range(self._first_epoch, epochs):
            self.epoch(epoch + 1, context)

            if self.stopped:
                break

        self.metrics.end_train()
        self._first_epoch = epochs

    def epoch(self, epoch, context):
        self.current_epoch = epoch
        self.metrics.new_epoch(epoch, context)
        iterations = len(self.dataloader) * (epoch - 1)

        for step, mini_batch in enumerate(self.dataloader):
            step += iterations
            self.metrics.new_batch(step, mini_batch, None)

            results = self.step(step, mini_batch, context)

            self.metrics.end_batch(step, mini_batch, results)

        self.metrics.end_epoch(epoch, context)

    def step(self, step, input, context):
        self.classifier.train()
        self.optimizer.zero_grad()

        batch, target = input

        batch = [x.to(device=self.device) for x in batch]
        predictions = self.classifier(*batch)
        loss = self.criterion(predictions, target.to(device=self.device))

        self.optimizer.backward(loss)
        self.optimizer.step()

        results = {
            # needed to compute online loss
            "loss": loss.detach(),
            # needed to compute only accuracy
            "predictions": predictions.detach(),
        }

        return results

    def load_state_dict(self, state, strict=True):
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state["epoch"]
        self.current_epoch = state["epoch"]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state["epoch"] = self.current_epoch
        return state

    def predict_scores(self, batch):
        with torch.no_grad():
            data = [x.to(device=self.device) for x in batch]
            return self.classifier(*data)

    def predict(self, batch, target=None):
        scores = self.predict_scores(batch)

        _, predicted = torch.max(scores, 1)

        loss = None
        if target is not None:
            loss = self.criterion(scores, target.to(device=self.device))

        return predicted, loss

    def accuracy(self, batch, target):
        """Compute the accuracy of a batch of samples using the current version of the model"""
        self.classifier.eval()

        with torch.no_grad():
            predicted, loss = self.predict(batch, target)
            acc = (predicted == target.to(device=self.device)).sum()

        self.classifier.train()
        return acc.float(), loss


def main(validate=False):
    """Build the configuration we want to train on the cluster"""
    from olympus.models import Model
    from olympus.datasets import DataLoader, Dataset, SplitDataset
    from olympus.optimizers import Optimizer
    from olympus.utils import fetch_device

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
    )

    # Some models can accept different input size
    input_size, target_size = loader.get_shapes()
    model = Model(
        "resnet18",
        input_size=input_size,
        output_size=target_size[0],
    )

    optimizer = Optimizer("sgd")

    train, valid, test = loader.get_loaders(hpo_done=False)
    metrics = []
    if validate and valid:
        metrics.append(Accuracy(name="validation", loader=valid))

    if validate and test:
        metrics.append(Accuracy(name="test", loader=test))

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

    trainer.fit(epochs=100)


if __name__ == "__main__":
    main()
