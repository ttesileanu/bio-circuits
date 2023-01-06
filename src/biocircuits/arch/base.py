"""Define a base for all online models."""

import torch
from torch import nn
from typing import Union, Dict, Optional


class BaseOnlineModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trainer = None
        self.logger = None
        self.sample_idx = 0
        self.batch_idx = 0

        self.optimizers = []

        self._logged_during_step = False

    def log(self, name: str, value: Union[None, float, int] = None):
        if self.logger is not None:
            self.logger.log(name, value)
            self._logged_during_step = True

    def training_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        self._logged_during_step = False
        out = self.training_step_impl(batch)

        if self._logged_during_step:
            self.log("sample", self.sample_idx)
            self.log("batch", self.batch_idx)

        self.batch_idx += 1
        self.sample_idx += len(batch)

        return out

    def test_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Run only inference part of step."""
        return self.test_step_impl(batch)

    def training_step_impl(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError("need to override training_step_impl method")

    def test_step_impl(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError("need to override test_step_impl method")
