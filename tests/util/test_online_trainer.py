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
def trainer(loader):
    net = SimpleNamespace(forward=Mock(return_value=torch.zeros(4)), backward=Mock())
    trainer = OnlineTrainer(net, loader)
    return trainer


@pytest.fixture
def net_optim():
    return SimpleNamespace(
        forward=Mock(return_value=torch.zeros(4)),
        backward=Mock(),
        configure_optimizers=Mock(return_value=([Mock()], [Mock()])),
    )


@pytest.fixture()
def sample_history(loader) -> SimpleNamespace:
    net = SimpleNamespace(
        forward=Mock(return_value=torch.FloatTensor([0.0, 0.0])), backward=Mock()
    )

    loader = generate_loader(dim=3, batch_size=4, n_batches=25)
    trainer = OnlineTrainer(net, loader)

    n_samples = 0
    for batch in trainer:
        batch.training_step()
        batch.log_batch("x", batch.data[0])
        n_samples += len(batch)

    return SimpleNamespace(
        history=trainer.logger.history, loader=loader, n_samples=n_samples
    )


def test_trainer_iterates_have_data_member(trainer):
    for batch in trainer:
        assert hasattr(batch, "data")


def test_iterating_through_trainer_yields_correct_data(trainer, loader):
    for batch, (x,) in zip(trainer, loader):
        assert torch.allclose(batch.data[0], x)


def test_trainer_call_returns_sequence_with_correct_len(trainer, loader):
    assert len(trainer) == len(loader)


def test_trainer_call_returns_sequence_with_right_no_of_elems(trainer, loader):
    data = [_ for _ in trainer]
    assert len(data) == len(loader)


def test_batch_training_step_calls_net_forward(trainer):
    batch = next(iter(trainer))
    batch.training_step()

    trainer.model.forward.assert_called_once()


def test_batch_training_step_calls_net_forward_with_data(trainer):
    batch = next(iter(trainer))
    batch.training_step()

    trainer.model.forward.assert_called_once_with(*batch.data)


def test_batch_training_step_returns_output_from_forward(trainer):
    ret_val = torch.FloatTensor([1.0, -1.0])
    trainer.model.forward.return_value = ret_val
    for batch in trainer:
        out = batch.training_step()

    assert torch.allclose(out, ret_val)


def test_batch_training_step_calls_net_backward_with_output_from_forward(trainer):
    batch = next(iter(trainer))
    out = batch.training_step()

    trainer.model.backward.assert_called_once_with(out)


def test_batch_contains_batch_index(trainer):
    for i, batch in enumerate(trainer):
        assert i == batch.idx


def test_batch_every(trainer):
    s = 3
    for batch in trainer:
        assert batch.every(s) == ((batch.idx % s) == 0)


def test_batch_count(trainer, loader):
    n = len(loader)
    c = 13
    idxs = np.linspace(0, n - 1, c).astype(int)
    for batch in trainer:
        assert batch.count(c) == (batch.idx in idxs)


def test_batch_len(trainer):
    for batch in trainer:
        assert len(batch) == len(batch.data[0])


def test_batch_keeps_track_of_sample_index(trainer):
    crt_sample = 0
    for batch in trainer:
        assert batch.sample_idx == crt_sample
        crt_sample += len(batch)


def test_batch_terminate_ends_iteration(trainer):
    count = 0
    n = 5
    for batch in trainer:
        count += 1
        if batch.idx == n - 1:
            batch.terminate()

    assert count == n


def test_batch_terminate_only_terminates_on_next_iter(trainer):
    count = 0
    for batch in trainer:
        batch.terminate()
        count += 1

    assert count == 1


def test_batch_log_fills_in_sample_index(trainer):
    expected = []
    for batch in trainer:
        batch.log("foo", 2.0)
        expected.append(batch.sample_idx)

    history = trainer.logger.history
    assert "foo.sample" in history

    samples = history["foo.sample"]
    assert len(samples) == len(expected)

    np.testing.assert_equal(samples, expected)


def test_end_of_iteration_calls_logger_finalize(trainer):
    for batch in trainer:
        batch.log("foo", 1.0)

    assert trainer.logger.finalized


def test_trainer_log_batch_reports_correct_samples(sample_history):
    expected_sample = np.arange(sample_history.n_samples)
    np.testing.assert_equal(sample_history.history["x.sample"], expected_sample)


def test_batch_accumulate(trainer):
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(trainer))

    expected = []
    for i in range(len(values) // 4):
        expected.append(np.mean(values[i * 4 : (i + 1) * 4]))

    calculated = []
    for batch in trainer:
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
def test_train_batch_repr(trainer, kind):
    batch = next(iter(trainer))

    s = {"repr": repr, "str": str}[kind](batch)
    assert s.startswith("TrainingBatch(")
    assert s.endswith(")")


def test_multiparam_log_stores_sample_for_each_key(trainer):
    for batch in trainer:
        batch.log({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_batch_stores_sample_for_each_key(trainer):
    a = np.array([0.0, 1.0])
    for batch in trainer:
        batch.log_batch({"foo": a, "bar": a})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_accumulated_stores_sample_for_each_key(trainer):
    for batch in trainer:
        if batch.every(4):
            batch.log_accumulated()
        batch.accumulate({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_log_with_index_prefix(trainer):
    for batch in trainer:
        batch.log("foo", 0, index_prefix="bar")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_with_index_prefix(trainer):
    for batch in trainer:
        batch.log({"foo": 0.0, "bar": 1.0}, index_prefix="boo")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_log_batch_with_index_prefix(trainer):
    a = np.array([0.0, 1.0])
    for batch in trainer:
        batch.log_batch("foo", a, index_prefix="bar")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" in trainer.logger.history


def test_multiparam_log_batch_with_index_prefix(trainer):
    a = np.array([0.0, 1.0])
    for batch in trainer:
        batch.log_batch({"foo": a, "bar": a}, index_prefix="boo")

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_log_accumulated_with_index_prefix(trainer):
    for batch in trainer:
        if batch.every(4):
            batch.log_accumulated(index_prefix="boo")
        batch.accumulate({"foo": 0.0, "bar": 1.0})

    assert "foo.sample" not in trainer.logger.history
    assert "bar.sample" not in trainer.logger.history
    assert "boo.sample" in trainer.logger.history


def test_batch_training_step_calls_model_training_step_if_available(trainer):
    trainer.model.training_step = Mock()

    batch = next(iter(trainer))
    batch.training_step()

    trainer.model.forward.assert_not_called()
    trainer.model.backward.assert_not_called()
    trainer.model.training_step.assert_called()


def test_batch_training_step_calls_model_training_step_with_batch(trainer):
    trainer.model.training_step = Mock()

    for batch in trainer:
        batch.training_step()
        trainer.model.training_step.assert_called_with(batch)


def test_cannot_iterate_twice(trainer):
    for _ in trainer:
        pass

    with pytest.raises(IndexError):
        for _ in trainer:
            pass


def test_trainer_init_calls_model_configure_optimizers(net_optim, loader):
    trainer = OnlineTrainer(net_optim, loader)
    trainer.model.configure_optimizers.assert_called_once_with()


def test_trainer_init_passes_lr_to_configure_optimizers(net_optim, loader):
    lr = 0.123
    trainer = OnlineTrainer(net_optim, loader, lr=lr)
    trainer.model.configure_optimizers.assert_called_once_with(lr=lr)


def test_trainer_init_passes_optim_kws_to_configure_optimizers(net_optim, loader):
    trainer = OnlineTrainer(net_optim, loader, optim_kws={"foo": "bar"})
    trainer.model.configure_optimizers.assert_called_once_with(foo="bar")


def test_trainer_init_stores_output_from_configure_optimizers(net_optim, loader):
    optim = Mock()
    sched = Mock()
    net_optim.configure_optimizers.return_value = ([optim], [sched])
    trainer = OnlineTrainer(net_optim, loader)

    assert len(trainer.optimizers) == 1
    assert trainer.optimizers[0] is optim

    assert len(trainer.schedulers) == 1
    assert trainer.schedulers[0] is sched
