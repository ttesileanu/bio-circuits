import pytest
from unittest.mock import Mock, patch

import torch
import numpy as np

from types import SimpleNamespace

from biocircuits.util.online_trainer import OnlineTrainer
from biocircuits.log.logger import Logger
from biocircuits.arch.base import BaseOnlineModel
from biocircuits.util.callbacks import BaseCallback, LambdaCallback


class DefaultModel(BaseOnlineModel):
    def __init__(self):
        super().__init__()
        self.batch_idx = 0

    def training_step_impl(self, batch: torch.Tensor):
        self.batch_idx += 1

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


def test_fit_exits_if_checkpoint_returns_false(loader):
    model = Mock()
    checkpoint = Callback(model, "checkpoint", output=False)

    trainer = OnlineTrainer(callbacks=[checkpoint])
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
    checkpoint = Callback(model, "checkpoint", "post")

    trainer = OnlineTrainer(callbacks=[checkpoint])
    trainer.fit(model, loader[:1])

    assert checkpoint.batch_idx_at_call is not None
    # this call should be *after* the step
    assert checkpoint.batch_idx_at_call == 1


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


def test_fit_calls_logger_initialize(default_trainer, loader):
    model = DefaultModel()
    logger = default_trainer.logger
    with patch.object(
        logger, "initialize", wraps=logger.initialize
    ) as wrapped_initialize:
        default_trainer.fit(model, loader)
        wrapped_initialize.assert_called_once()


def test_fit_calls_logger_finalize(default_trainer, loader):
    model = DefaultModel()
    logger = default_trainer.logger
    with patch.object(logger, "finalize", wraps=logger.finalize) as wrapped_finalize:
        default_trainer.fit(model, loader)
        wrapped_finalize.assert_called_once()


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


@pytest.mark.parametrize("scope", ["training", "both"])
def test_fit_calls_callback_initialize_for_scope(scope, loader):
    callback = Mock(intent="checkpoint", timing="post", scope=scope)
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)
    callback.initialize.assert_called()


def test_fit_does_not_call_callback_initialize_for_scope_test(loader):
    callback = Mock(intent="checkpoint", timing="post", scope="test")
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)
    callback.initialize.assert_not_called()


@pytest.mark.parametrize("scope", ["training", "both"])
def test_fit_calls_callback_finalize_for_scope(scope, loader):
    callback = Mock(intent="checkpoint", timing="post", scope=scope)
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)
    callback.finalize.assert_called()


def test_fit_does_not_call_callback_finalize_for_scope_test(loader):
    callback = Mock(intent="checkpoint", timing="post", scope="test")
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)
    callback.finalize.assert_not_called()


@pytest.mark.parametrize("scope", ["test", "both"])
def test_predict_calls_callback_initialize_for_scope(scope, loader):
    callback = Mock(intent="checkpoint", timing="post", scope=scope)
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)
    callback.initialize.assert_called()


def test_predict_does_not_call_initialize_for_training(loader):
    callback = Mock(intent="checkpoint", timing="post", scope="training")
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)
    callback.initialize.assert_not_called()


@pytest.mark.parametrize("scope", ["test", "both"])
def test_predict_calls_callback_finalize_for_scope(scope, loader):
    callback = Mock(intent="checkpoint", timing="post", scope=scope)
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)
    callback.finalize.assert_called()


def test_predict_does_not_call_finalize_for_training(loader):
    callback = Mock(intent="checkpoint", timing="post", scope="training")
    model = DefaultModel()

    trainer = OnlineTrainer(callbacks=[callback])
    trainer.predict(model, loader)
    callback.finalize.assert_not_called()


def test_fit_sets_max_batches(default_trainer, loader):
    model = DefaultModel()
    default_trainer.fit(model, loader)
    assert default_trainer.max_batches == len(loader)


def test_predict_does_not_set_max_batches(default_trainer, loader):
    model = DefaultModel()
    default_trainer.max_batches = -1
    default_trainer.predict(model, loader)
    assert default_trainer.max_batches == -1


def test_fit_keeps_track_of_batch_index(loader):
    indices = []
    callback = LambdaCallback(
        lambda _, trainer, storage=indices: storage.append(trainer.batch_idx) or True,
        timing="pre",
    )
    model = DefaultModel()
    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)

    assert indices == list(range(len(loader)))


def test_fit_keeps_track_of_sample_index(loader):
    indices = []
    callback = LambdaCallback(
        lambda _, trainer, storage=indices: storage.append(trainer.sample_idx) or True,
        timing="pre",
    )
    model = DefaultModel()
    trainer = OnlineTrainer(callbacks=[callback])
    trainer.fit(model, loader)

    batch_size = len(loader[0])
    assert indices == list(range(0, batch_size * len(loader), batch_size))


def test_fit_sends_model_for_progress_to_progress_callback(loader):
    model = DefaultModel()
    model.for_progress = {"foo": 3, "bar": 5}
    progress = Mock(intent="progress", timing="post", scope="training")

    trainer = OnlineTrainer(callbacks=[progress])
    trainer.fit(model, loader[:1])

    progress.assert_called_with(model.for_progress)


def test_predict_sends_model_for_progress_to_progress_callback(loader):
    model = DefaultModel()
    model.for_progress = {"foo": 3, "bar": 5}
    progress = Mock(intent="progress", timing="post", scope="test")

    trainer = OnlineTrainer(callbacks=[progress])
    trainer.predict(model, loader[:1])

    progress.assert_called_with(model.for_progress)
