import pytest

import torch

from unittest.mock import Mock, patch
from torch import nn
from biocircuits.arch.base import BaseOnlineModel
from typing import Optional


class NopModel(BaseOnlineModel):
    def __init__(self):
        super().__init__()
        self.training_impl_called = False
        self.test_impl_called = False

    def training_step_impl(self, batch):
        self.training_impl_called = True

    def test_step_impl(self, batch):
        self.test_impl_called = True


@pytest.fixture
def model():
    res = NopModel()
    return res


def test_calling_training_step_on_base_class_raises():
    base = BaseOnlineModel()
    with pytest.raises(NotImplementedError):
        base.training_step([])


def test_calling_test_step_on_base_class_raises():
    base = BaseOnlineModel()
    with pytest.raises(NotImplementedError):
        base.test_step([])


def test_training_step_calls_impl(model):
    model.training_step([])
    assert model.training_impl_called


def test_test_step_calls_impl(model):
    model.test_step([])
    assert model.test_impl_called


def test_inherits_from_module(model):
    assert isinstance(model, nn.Module)


def test_trainer_init_to_none(model):
    assert model.trainer is None


def test_logger_init_to_none(model):
    assert model.logger is None


def test_sample_idx_starts_at_zero(model):
    assert model.sample_idx == 0


def test_batch_idx_starts_at_zero(model):
    assert model.batch_idx == 0


def test_log_calls_logger_log(model):
    logger = Mock()
    model.logger = logger

    name = "foo"
    value = 3.0
    model.log(name, value)

    logger.log.assert_called_once_with(name, value)


def test_training_step_increments_batch_idx(model):
    model.training_step([])
    assert model.batch_idx == 1


def test_training_step_increases_sample_idx_by_batch_len(model):
    model.training_step([1, 2, 3])
    assert model.sample_idx == 3


def test_test_step_does_not_change_batch_or_sample_idx(model):
    model.test_step([])
    assert model.batch_idx == 0
    assert model.sample_idx == 0


def test_training_step_logs_nothing_by_default(model):
    logger = Mock()
    model.logger = logger
    model.training_step([])
    logger.log.assert_not_called()


def test_test_step_logs_nothing_by_default(model):
    logger = Mock()
    model.logger = logger
    model.test_step([])
    logger.log.assert_not_called()


class LoggingModel(BaseOnlineModel):
    def training_step_impl(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        self.log("foo", 1.0)


def test_training_step_logs_indices_if_anything_was_logged_since_last_step():

    logger = Mock()
    model = LoggingModel()
    model.logger = logger

    sample = 5
    batch = 2
    model.sample_idx = sample
    model.batch_idx = batch

    model.training_step([])
    logger.log.assert_any_call("sample", sample)
    logger.log.assert_any_call("batch", batch)


def test_log_fails_silently_if_logger_is_not_set(model):
    model.log("key", 3.0)


def test_init_creates_empty_optimizer_list(model):
    assert len(model.optimizers) == 0


def test_log_uses_trainer_logger_if_logger_is_none(model):
    trainer = Mock()
    model.trainer = trainer
    model.log("key", 3.0)

    trainer.logger.log.assert_called_once()
