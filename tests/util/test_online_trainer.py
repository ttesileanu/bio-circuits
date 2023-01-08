import pytest
from unittest.mock import Mock, MagicMock

import torch
import numpy as np

from types import SimpleNamespace

from biocircuits.util.online_trainer import OnlineTrainer
from biocircuits.log.logger import Logger
from biocircuits.arch.base import BaseOnlineModel
from biocircuits.util.callbacks import BaseCallback


class DefaultModel(BaseOnlineModel):
    def training_step_impl(self, batch: torch.Tensor):
        pass

    def test_step_impl(self, batch: torch.Tensor):
        pass


class ModelWithOutput(BaseOnlineModel):
    def training_step_impl(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.FloatTensor([1.0, 2.0, 3.0])

    def test_step_impl(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.FloatTensor([1.0, 2.0, -3.0])


class Callback(BaseCallback):
    def __init__(self, model, *args, output: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.output = output
        self.batch_idx_at_call = None

    def __call__(self, *args, **kwargs):
        self.batch_idx_at_call = self.model.batch_idx
        return self.output


@pytest.fixture
def default_trainer() -> OnlineTrainer:
    return OnlineTrainer()


def generate_loader(dim: int, batch_size: int, n_batches: int):
    torch.manual_seed(321)
    data = [(torch.randn((batch_size, dim)),) for _ in range(n_batches)]
    return data


@pytest.fixture
def loader():
    return generate_loader(dim=3, batch_size=4, n_batches=25)


def test_default_callbacks_is_empty(default_trainer):
    assert len(default_trainer.callbacks) == 0


def test_init_creates_new_logger_by_default(default_trainer):
    assert isinstance(default_trainer.logger, Logger)


def test_fit_sets_model_trainer_attribute(default_trainer, loader):
    model = DefaultModel()
    default_trainer.fit(model, loader)
    assert model.trainer is default_trainer


def test_predict_sets_model_trainer_attribute(default_trainer, loader):
    model = DefaultModel()
    default_trainer.predict(model, loader)
    assert model.trainer is default_trainer


def test_fit_calls_training_step_the_right_number_of_times(default_trainer, loader):
    model = Mock()
    default_trainer.fit(model, loader)
    assert model.training_step.call_count == len(loader)


def test_fit_calls_progress_callback_appropriately(loader):
    model = DefaultModel()
    progress = Callback(model, "progress")

    trainer = OnlineTrainer(callbacks=[progress])
    trainer.fit(model, loader[:1])

    assert progress.batch_idx_at_call is not None
    # by default call is after the step
    assert progress.batch_idx_at_call == 1


def test_fit_calls_checkpoint_callback_appropriately(loader):
    model = DefaultModel()
    checkpoint = Callback(model, "checkpoint")

    trainer = OnlineTrainer(callbacks=[checkpoint])
    trainer.fit(model, loader[:1])

    assert checkpoint.batch_idx_at_call is not None
    # by default call is after the step
    assert checkpoint.batch_idx_at_call == 1


def test_fit_calls_monitor_callback_appropriately(loader):
    model = DefaultModel()
    monitor = Callback(model, "monitor")

    trainer = OnlineTrainer(callbacks=[monitor])
    trainer.fit(model, loader[:1])

    assert monitor.batch_idx_at_call is not None
    # by default call is after the step
    assert monitor.batch_idx_at_call == 1


def test_fit_exits_if_monitor_returns_false(loader):
    model = Mock()
    monitor = Callback(model, "monitor", output=False)

    trainer = OnlineTrainer(callbacks=[monitor])
    trainer.fit(model, loader)

    assert len(loader) > 1
    assert model.training_step.call_count == 1


def test_fit_runs_callback_before_step_if_timing_is_pre(loader):
    model = DefaultModel()
    checkpoint = Callback(model, "checkpoint", "pre")

    trainer = OnlineTrainer(callbacks=[checkpoint])
    trainer.fit(model, loader[:1])

    assert checkpoint.batch_idx_at_call is not None
    # this call should be *before* the step
    assert checkpoint.batch_idx_at_call == 0


def test_fit_runs_callback_after_step_if_timing_is_post(loader):
    model = DefaultModel()
    monitor = Callback(model, "monitor", "post")

    trainer = OnlineTrainer(callbacks=[monitor])
    trainer.fit(model, loader[:1])

    assert monitor.batch_idx_at_call is not None
    # this call should be *after* the step
    assert monitor.batch_idx_at_call == 1


def test_predict_calls_test_step_the_right_number_of_times(default_trainer, loader):
    model = Mock()
    default_trainer.predict(model, loader)
    assert model.test_step.call_count == len(loader)


def test_fit_does_not_call_callback_if_scope_is_test(loader):
    model = DefaultModel()
    callback = Callback(model, scope="test")

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)

    assert callback.batch_idx_at_call is None


@pytest.mark.parametrize("scope", ["training", "both"])
def test_fit_calls_callback_if_scope_is(scope, loader):
    model = DefaultModel()
    callback = Callback(model, scope=scope)

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)

    assert callback.batch_idx_at_call is not None


def test_predict_does_not_call_callback_if_scope_is_training(loader):
    model = DefaultModel()
    callback = Callback(model, scope="training")

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)

    assert callback.batch_idx_at_call is None


@pytest.mark.parametrize("scope", ["test", "both"])
def test_predict_calls_callback_if_scope_is(scope, loader):
    model = DefaultModel()
    callback = Callback(model, scope=scope)

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)

    assert callback.batch_idx_at_call is not None


def test_fit_calls_logger_finalize_at_end_by_default(default_trainer, loader):
    model = DefaultModel()
    default_trainer.fit(model, loader)
    assert default_trainer.logger.finalized


def test_fit_can_disable_calling_logger_finalize(default_trainer, loader):
    model = DefaultModel()
    default_trainer.fit(model, loader, finalize_logger=False)
    assert not default_trainer.logger.finalized


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_trainer_repr(default_trainer, kind):
    s = {"repr": repr, "str": str}[kind](default_trainer)
    assert s.startswith("OnlineTrainer(")
    assert s.endswith(")")


def test_fit_returns_list_of_outputs(default_trainer, loader):
    model = ModelWithOutput()
    output = default_trainer.fit(model, loader)
    assert output is not None
    assert len(output) == len(loader)


def test_fit_returns_none_when_training_step_returns_none(default_trainer, loader):
    model = DefaultModel()
    output = default_trainer.fit(model, loader)
    assert output is None


def test_predict_returns_list_of_outputs(default_trainer, loader):
    model = ModelWithOutput()
    output = default_trainer.predict(model, loader)
    assert output is not None
    assert len(output) == len(loader)


def test_predict_returns_none_when_training_step_returns_none(default_trainer, loader):
    model = DefaultModel()
    output = default_trainer.predict(model, loader)
    assert output is None
