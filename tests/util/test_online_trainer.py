import pytest
from unittest.mock import Mock

import torch
import numpy as np

from types import SimpleNamespace

from biocircuits.util.online_trainer import OnlineTrainer


def generate_loader(dim: int, batch_size: int, n_batches: int):
    torch.manual_seed(321)
    data = [(torch.randn((batch_size, dim)),) for _ in range(n_batches)]
    return data


@pytest.fixture
def loader():
    return generate_loader(dim=3, batch_size=4, n_batches=25)


@pytest.fixture
def net():
    n = SimpleNamespace(forward=Mock(return_value=torch.zeros(4)), backward=Mock())
    return n


@pytest.fixture
def trainer():
    trainer = OnlineTrainer()
    return trainer


def test_trainer_iterates_have_data_member(trainer, net, loader):
    for batch in trainer(net, loader):
        assert hasattr(batch, "data")


def test_iterating_through_trainer_yields_correct_data(trainer, net, loader):
    for batch, (x,) in zip(trainer(net, loader), loader):
        assert torch.allclose(batch.data[0], x)


def test_trainer_call_returns_sequence_with_correct_len(trainer, net, loader):
    assert len(trainer(net, loader)) == len(loader)


def test_trainer_call_returns_sequence_with_right_no_of_elems(trainer, net, loader):
    data = [_ for _ in trainer(net, loader)]
    assert len(data) == len(loader)


def test_batch_training_step_calls_net_forward(trainer, net, loader):
    batch = next(iter(trainer(net, loader)))
    batch.training_step()

    net.forward.assert_called_once()


def test_batch_training_step_calls_net_forward_with_data(trainer, net, loader):
    batch = next(iter(trainer(net, loader)))
    batch.training_step()

    net.forward.assert_called_once_with(*batch.data)


def test_batch_training_step_returns_output_from_forward(trainer, net, loader):
    ret_val = torch.FloatTensor([1.0, -1.0])
    net.forward.return_value = ret_val
    for batch in trainer(net, loader):
        out = batch.training_step()

    assert torch.allclose(out, ret_val)


def test_batch_training_step_calls_net_backward_with_output_from_forward(
    trainer, net, loader
):
    batch = next(iter(trainer(net, loader)))
    out = batch.training_step()

    net.backward.assert_called_once_with(out)


def test_batch_contains_batch_index(trainer, net, loader):
    for i, batch in enumerate(trainer(net, loader)):
        assert i == batch.idx


def test_batch_every(trainer, net, loader):
    s = 3
    for batch in trainer(net, loader):
        assert batch.every(s) == ((batch.idx % s) == 0)


def test_batch_count(trainer, net, loader):
    n = len(loader)
    c = 13
    idxs = np.linspace(0, n - 1, c).astype(int)
    for batch in trainer(net, loader):
        assert batch.count(c) == (batch.idx in idxs)


def test_batch_len(trainer, net, loader):
    for batch in trainer(net, loader):
        assert len(batch) == len(batch.data[0])


def test_batch_keeps_track_of_sample_index(trainer, net, loader):
    crt_sample = 0
    for batch in trainer(net, loader):
        assert batch.sample_idx == crt_sample
        crt_sample += len(batch)


def test_batch_terminate_ends_iteration(trainer, net, loader):
    count = 0
    n = 5
    for batch in trainer(net, loader):
        count += 1
        if batch.idx == n - 1:
            batch.terminate()

    assert count == n


def test_batch_terminate_only_terminates_on_next_iter(trainer, net, loader):
    count = 0
    for batch in trainer(net, loader):
        batch.terminate()
        count += 1

    assert count == 1


def test_batch_log_fills_in_sample_index(trainer, net, loader):
    expected = []
    for batch in trainer(net, loader):
        batch.log("foo", 2.0)
        expected.append(batch.sample_idx)

    history = trainer.logger.history
    assert "foo.sample" in history

    samples = history["foo.sample"]
    assert len(samples) == len(expected)

    np.testing.assert_equal(samples, expected)


def test_end_of_iteration_calls_logger_finalize(trainer, net, loader):
    for batch in trainer(net, loader):
        batch.log("foo", 1.0)

    assert trainer.logger.finalized


@pytest.fixture()
def sample_history() -> SimpleNamespace:
    trainer = OnlineTrainer()
    loader = generate_loader(dim=3, batch_size=4, n_batches=25)

    net = Mock()
    net.forward.return_value = torch.FloatTensor([0.0, 0.0])

    n_samples = 0
    for batch in trainer(net, loader):
        batch.training_step()
        batch.log_batch("x", batch.data[0])
        n_samples += len(batch)

    return SimpleNamespace(
        history=trainer.logger.history, loader=loader, n_samples=n_samples
    )


def test_trainer_log_batch_reports_correct_samples(sample_history):
    expected_sample = np.arange(sample_history.n_samples)
    np.testing.assert_equal(sample_history.history["x.sample"], expected_sample)


def test_batch_accumulate(trainer, net, loader):
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(loader))

    expected = []
    for i in range(len(values) // 4):
        expected.append(np.mean(values[i * 4 : (i + 1) * 4]))

    calculated = []
    for batch in trainer(net, loader):
        if batch.every(4):
            calculated.append(batch.calculate_accumulated("foo"))
            batch.log_accumulated()
        batch.accumulate("foo", values[batch.idx])

    history = trainer.logger.history
    assert "foo" in history
    assert np.isnan(calculated[0])  # because nothing was accumulated yet
    np.testing.assert_allclose(calculated[1:], history["foo"])
    np.testing.assert_allclose(calculated[1:], expected)


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_trainer_repr(trainer, kind):
    s = {"repr": repr, "str": str}[kind](trainer)
    assert s.startswith("OnlineTrainer(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_train_iterable_repr(trainer, net, loader, kind):
    train_iterable = trainer(net, loader)

    s = {"repr": repr, "str": str}[kind](train_iterable)
    assert s.startswith("TrainingIterable(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_train_batch_repr(trainer, net, loader, kind):
    batch = next(iter(trainer(net, loader)))

    s = {"repr": repr, "str": str}[kind](batch)
    assert s.startswith("TrainingBatch(")
    assert s.endswith(")")


def test_multiparam_log_stores_sample_for_each_key(trainer, net, loader):
    for batch in trainer(net, loader):
        batch.log({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_batch_stores_sample_for_each_key(trainer, net, loader):
    a = np.array([0.0, 1.0])
    for batch in trainer(net, loader):
        batch.log_batch({"foo": a, "bar": a})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_accumulated_stores_sample_for_each_key(trainer, net, loader):
    for batch in trainer(net, loader):
        if batch.every(4):
            batch.log_accumulated()
        batch.accumulate({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_log_with_index_prefix(trainer, net, loader):
    for batch in trainer(net, loader):
        batch.log("foo", 0, index_prefix="bar")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_with_index_prefix(trainer, net, loader):
    for batch in trainer(net, loader):
        batch.log({"foo": 0.0, "bar": 1.0}, index_prefix="boo")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_log_batch_with_index_prefix(trainer, net, loader):
    a = np.array([0.0, 1.0])
    for batch in trainer(net, loader):
        batch.log_batch("foo", a, index_prefix="bar")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_batch_with_index_prefix(trainer, net, loader):
    a = np.array([0.0, 1.0])
    for batch in trainer(net, loader):
        batch.log_batch({"foo": a, "bar": a}, index_prefix="boo")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_log_accumulated_with_index_prefix(trainer, net, loader):
    for batch in trainer(net, loader):
        if batch.every(4):
            batch.log_accumulated(index_prefix="boo")
        batch.accumulate({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_batch_training_step_calls_model_training_step_if_available(
    net, trainer, loader
):
    net.training_step = Mock()

    batch = next(iter(trainer(net, loader)))
    batch.training_step()

    net.forward.assert_not_called()
    net.backward.assert_not_called()
    net.training_step.assert_called()


def test_batch_training_step_calls_model_training_step_with_batch(net, trainer, loader):
    net.training_step = Mock()

    for batch in trainer(net, loader):
        batch.training_step()
        net.training_step.assert_called_with(batch)
