"""Define a base for all online models."""

import torch
from torch import nn
from typing import Union, Dict, Optional


class BaseOnlineModel(nn.Module):
    """Base for online training model.

    Attributes:
        trainer : util.OnlineTrainer
            Trainer object used with the model.
        logger : Optional[util.Logger]
            Logger object. If it is `None`, `trainer.logger` is used.
        sample_idx : int
            Number of training samples processed.
        batch_idx : int
            Number of training batches processed.
        optimizers : List[torch.optim.Optimizer]
            List of optimizers used by the model.
    """

    def __init__(self, *args, **kwargs):
        """Initialize base online model.

        All arguments are sent to `nn.Module`.
        """
        super().__init__(*args, **kwargs)

        self.trainer = None
        self.logger = None
        self.sample_idx = 0
        self.batch_idx = 0

        self.optimizers = []

        self._logged_during_step = False

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

        self._logged_during_step = True

    def training_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Run a training step.

        This calls `self.training_step_impl` for the actual implementation of the
        training. After the training step it logs the current sample and batch indices
        (but only if `self.log` has been called in `training_step_impl`). It then
        proceeds to increment these indices appropriately.
        """
        self._logged_during_step = False
        out = self.training_step_impl(batch)

        if self._logged_during_step:
            self.log("sample", self.sample_idx)
            self.log("batch", self.batch_idx)

        self.batch_idx += 1
        self.sample_idx += len(batch)

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
