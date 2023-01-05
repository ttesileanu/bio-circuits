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

        self._logged_since_step = False

    def log(self, name: str, value: Union[None, float, int] = None):
        if self.logger is not None:
            self.logger.log(name, value)
            self._logged_since_step = True

    def training_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        if self._logged_since_step:
            self.log("sample", self.sample_idx)
            self.log("batch", self.batch_idx)

        self._logged_since_step = False

        self.batch_idx += 1
        self.sample_idx += len(batch)

    def test_step(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Run only inference part of step."""
        pass
