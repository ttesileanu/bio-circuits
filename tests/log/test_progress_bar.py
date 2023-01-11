import pytest
import tqdm

from unittest.mock import Mock

from biocircuits.log.progress_bar import ProgressBar
from biocircuits.util.callbacks import BaseCallback


@pytest.fixture
def bar() -> ProgressBar:
    return ProgressBar()


def test_model_checkpoint_inherits_from_base_callback(bar):
    assert isinstance(bar, BaseCallback)


def test_intent_is_progress(bar):
    assert bar.intent == "progress"


def test_timing_is_post(bar):
    assert bar.timing == "post"


def test_default_scope_is_training(bar):
    assert bar.scope == "training"


def test_initialize_calls_backend_with_total_equal_to_trainer_max_batches():
    backend = Mock()
    progress = ProgressBar(backend=backend)

    max_batches = 23
    trainer = Mock(max_batches=max_batches)
    progress.initialize(trainer)
    backend.assert_called()
    assert "total" in backend.call_args.kwargs
    assert backend.call_args.kwargs["total"] == max_batches


def test_finalize_calls_close_on_backend_instance():
    backend_instance = Mock()
    backend = Mock(return_value=backend_instance)
    progress = ProgressBar(backend=backend)
    progress.initialize(Mock())
    progress.finalize()
    backend_instance.close.assert_called()


def test_call_calls_update():
    backend_instance = Mock()
    backend = Mock(return_value=backend_instance)
    progress = ProgressBar(backend=backend)

    progress.initialize(Mock())
    progress({})

    backend_instance.update.assert_called_once_with(1)


def test_call_calls_set_postfix():
    backend_instance = Mock()
    backend = Mock(return_value=backend_instance)
    progress = ProgressBar(backend=backend)

    progress.initialize(Mock())
    d = {"foo": 3, "bar": 5}
    progress(d)

    backend_instance.set_postfix.assert_called_once_with(d)


def test_default_backend_is_tqdm(bar):
    assert bar.backend == tqdm.tqdm


def test_additional_kwargs():
    backend = Mock()
    progress = ProgressBar(backend=backend, foo=5)

    progress.initialize(Mock())
    backend.assert_called()
    assert "foo" in backend.call_args.kwargs
    assert backend.call_args.kwargs["foo"] == 5
