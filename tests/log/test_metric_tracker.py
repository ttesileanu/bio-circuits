import pytest

from unittest.mock import Mock
from biocircuits.log.metric_tracker import MetricTracker


# MetricTracker({"loss": lambda model, _: model.loss(test)})


@pytest.fixture
def tracker():
    return MetricTracker()


def test_intent_is_checkpoint(tracker):
    assert tracker.intent == "checkpoint"


def test_default_timing_is_post(tracker):
    assert tracker.timing == "post"


def test_default_scope_is_training(tracker):
    assert tracker.scope == "training"


def test_default_frequency_is_one(tracker):
    assert tracker.frequency == 1


def test_tracking_for_log_but_not_report_progress():
    tracker = MetricTracker(
        log={"foobar": lambda model, trainer: model.foo * trainer.bar}
    )

    foo = 3.5
    bar = -0.7
    model = Mock(foo=foo)
    trainer = Mock(bar=bar, batch_idx=0)

    tracker(model, trainer)
    model.log.assert_called_once_with("foobar", foo * bar)
    model.report_progress.assert_not_called()


def test_tracking_returns_true():
    tracker = MetricTracker({"foo": lambda *args: 1.0})
    assert tracker(Mock(), Mock(batch_idx=0))


def test_tracking_for_progress_only():
    tracker = MetricTracker(progress={"foo": lambda *args: 1.0})
    model = Mock()
    tracker(model, Mock(batch_idx=0))

    model.report_progress.assert_called_once_with("foo", 1.0)
    model.log.assert_not_called()


def test_tracking_for_both_log_and_progress():
    tracker = MetricTracker(both={"foo": lambda *args: 1.0})
    model = Mock()
    tracker(model, Mock(batch_idx=0))

    model.report_progress.assert_called_once_with("foo", 1.0)
    model.log.assert_called_once_with("foo", 1.0)


def test_default_tracking_is_for_both():
    tracker = MetricTracker({"foo": lambda *args: 1.0})
    model = Mock()
    tracker(model, Mock(batch_idx=0))

    model.report_progress.assert_called_once_with("foo", 1.0)
    model.log.assert_called_once_with("foo", 1.0)


def test_frequency_obeyed():
    frequency = 3
    tracker = MetricTracker({"foo": lambda *args: 1.0}, frequency=frequency)

    model = Mock()
    trainer = Mock()
    for i in range(10):
        trainer.batch_idx = i

        prev_call_count = model.log.call_count
        tracker(model, trainer)
        if i % frequency == 0:
            assert model.log.call_count == prev_call_count + 1
        else:
            assert model.log.call_count == prev_call_count
