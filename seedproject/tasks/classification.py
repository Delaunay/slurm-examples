import torch
from olympus.metrics import OnlineTrainAccuracy
from olympus.observers import (
    CheckPointer,
    ElapsedRealTime,
    ProgressView,
    SampleCount,
    Speed,
)
from olympus.resuming import load_state_dict, state_dict
from olympus.tasks.task import Task
from olympus.utils import drop_empty_key, select
from torch.nn import CrossEntropyLoss


class Classification(Task):
    """This reimplements a simplified version of olympus.tasks.classification as an example"""

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
            for metric in metrics:
                self.metrics.append(metric)

        self.metrics.append(ProgressView(self.metrics.get("Speed")))

        # Add checkpointing if we have a valid storage location
        if storage:
            self.metrics.append(CheckPointer(storage=storage))

        # Used when resuming
        self._first_epoch = 0
        self.current_epoch = 0
        self.hyper_parameters = dict()

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
        """Set our hyperparameters and instantiate the model and our optimizier"""

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
        # We will use this to generate a unique ID that will be used
        # to identify our current training.
        hyperparameters = self.get_current_space()

        # Trial Creation and Trial resume
        self.metrics.new_trial(hyperparameters, uid)
        self.set_device(self.device)

    def fit(self, epochs, context=None):
        """Train the current model"""
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
        """Iterate through the dataset once"""

        self.current_epoch = epoch
        self.metrics.new_epoch(epoch, context)

        for step, mini_batch in enumerate(self.dataloader):
            self.metrics.new_batch(step, mini_batch, None)

            results = self.step(step, mini_batch, context)

            self.metrics.end_batch(step, mini_batch, results)

        self.metrics.end_epoch(epoch, context)

    def step(self, step, batch, context):
        """Do a single optimizer step"""

        self.classifier.train()
        self.optimizer.zero_grad()

        obs, target = batch

        obs = [obs.to(device=self.device) for x in obs]
        predictions = self.classifier(*obs)
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
        """Load a state to resume training"""
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state["epoch"]
        self.current_epoch = state["epoch"]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save the current state of the training"""
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state["epoch"] = self.current_epoch
        return state

    def predict_scores(self, batch):
        """Compute the prediction score (value between 0-1) that gives the network confidence
        about the observation and its class"""
        with torch.no_grad():
            data = [x.to(device=self.device) for x in batch]
            return self.classifier(*data)

    def predict(self, batch, target=None):
        """Returns the class witht he highest confidence"""
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
