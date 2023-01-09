import pytest

from unittest.mock import Mock

from biocircuits.log.model_checkpoint import ModelCheckpoint
from biocircuits.util.callbacks import BaseCallback


@pytest.fixture
def checkpoint() -> ModelCheckpoint:
    return ModelCheckpoint()


def test_model_checkpoint_inherits_from_base_callback(checkpoint):
    assert isinstance(checkpoint, BaseCallback)


def test_intent_is_checkpoint(checkpoint):
    assert checkpoint.intent == "checkpoint"


def test_default_timing_is_post(checkpoint):
    assert checkpoint.timing == "post"


def test_default_scope_is_training(checkpoint):
    assert checkpoint.scope == "training"


def test_checkpoints_initially_empty(checkpoint):
    assert len(checkpoint.checkpoints) == 0


def test_copies_model_state_dict(checkpoint):
    model = Mock(batch_idx=0)

    state_dict = {"foo": [2.0, 3]}
    model.state_dict.return_value = state_dict

    checkpoint(model)

    assert len(checkpoint.checkpoints) == 1

    state_dict["foo"][0] = -2.0
    assert checkpoint.checkpoints[0]["foo"][0] > 0


def test_raises_for_unknown_criterion():
    with pytest.raises(ValueError):
        ModelCheckpoint(criterion="foo")


def test_frequency_criterion():
    frequency = 12
    checkpoint = ModelCheckpoint(criterion="batch", frequency=frequency)

    model = Mock()

    # this shouldn't make checkpoint
    model.batch_idx = 1
    checkpoint(model)
    assert len(checkpoint.checkpoints) == 0

    # this should
    model.batch_idx = 2 * frequency
    checkpoint(model)
    assert len(checkpoint.checkpoints) == 1


def test_callable_criterion():
    checkpoint = ModelCheckpoint(criterion=lambda model: model.foo == 3)

    # this shouldn't make checkpoint
    model = Mock(foo=5)
    checkpoint(model)
    assert len(checkpoint.checkpoints) == 0

    # this should
    model.foo = 3
    checkpoint(model)
    assert len(checkpoint.checkpoints) == 1


def test_stores_indices():
    frequency = 3
    checkpoint = ModelCheckpoint(criterion="batch", frequency=frequency)

    sample_size = 5
    max_batches = 10
    model = Mock()
    for batch_idx in range(max_batches):
        model.sample_idx = batch_idx * sample_size
        model.batch_idx = batch_idx
        checkpoint(model)

    expected_batches = list(range(0, max_batches, frequency))
    expected_samples = [_ * sample_size for _ in expected_batches]

    indices = checkpoint.indices
    assert "batch_idx" in indices
    assert "sample_idx" in indices

    assert indices["batch_idx"] == expected_batches
    assert indices["sample_idx"] == expected_samples
