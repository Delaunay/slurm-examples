import os
import logging
import tempfile

from seedproject.distributed.distributed as dist


log = logging.getLogger()


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
            dist.barrier()
            return

        # wait for rank 0 to save the checkpoint
        dist.barrier()

        # the other workers need to load the checkpoint
        if not dist.has_weight_autority():
            # this is made to make it work with a single GPU
            # with 2 processes on a single GPU
            # for testing purposes
            map_location = {"cuda:%d" % 0: "cuda:%d" % dist.device_id()}

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

        # wait for everybody to load the checkpoint
        dist.barrier()

    def save_checkpoint(self):
        """Save the current state of the trained to make it resumable"""
        log.info("save checkpoint")

        # only rank 0 can save the model
        if dist.has_weight_autority():
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
