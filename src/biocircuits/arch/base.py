"""Define a base for all online models."""

import torch
from torch import nn
from typing import Union, Dict, Optional, Any


class BaseOnlineModel(nn.Module):
    """Base for online training model.

    Attributes:
        trainer : util.OnlineTrainer
            Trainer object used with the model.
        logger : Optional[util.Logger]
            Logger object. If it is `None`, `trainer.logger` is used.
        optimizers : List[torch.optim.Optimizer]
            List of optimizers used by the model.
        for_progress: Dict[str, Any]
            Dictionary used for progress reports.
    """

    def __init__(self, *args, **kwargs):
        """Initialize base online model.

        All arguments are sent to `nn.Module`.
        """
        super().__init__(*args, **kwargs)

        self.trainer = None
        self.logger = None
        self.for_progress = {}

        self.optimizers = []

    def log(self, name: str, value: Union[None, float, int] = None):
        """Log a scalar value.

        This uses `self.logger` if it exists, otherwise tries `self.trainer.logger`.

        :param name: key to associate the value to
        :param value: value to log
        """
        if self.logger is not None:
            logger = self.logger
        elif self.trainer is not None:
            logger = self.trainer.logger
        else:
            logger = None

        if logger is not None:
            logger.log(name, value)

    def report_progress(self, key: str, value: Any):
        """Report a key-value pair for use with a progress bar.

        The `value` is simply stored in `self.for_progress`, replacing any previous
        stored value for the given `key`. Values are never removed automatically, but
        they can be removed manually using `self.unreport()`.
        """
        self.for_progress[key] = value

    def unreport(self, key: str) -> bool:
        """Remove the `key` from the progress bar.

        If the `key` is not found in `self.for_progress`, the function simply returns
        `False`.
        """
        if key in self.for_progress:
            self.for_progress.pop(key)
            return True
        else:
            return False

    def training_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Run a training step.

        This calls `self.training_step_impl` for the actual implementation of the
        training. After the training step it logs the current sample and batch indices
        (but only if `self.log` has been called in `training_step_impl`). It then
        proceeds to increment these indices appropriately.
        """
        out = self.training_step_impl(batch)
        if self.logger is not None:
            self.logger.step()
        elif self.trainer is not None:
            self.trainer.logger.step()

        return out

    def test_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Run only inference part of step.

        This calls `self.test_step_impl` for the actual implementation of the test step.
        """
        return self.test_step_impl(batch)

    def training_step_impl(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError("need to override training_step_impl method")

    def test_step_impl(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError("need to override test_step_impl method")
