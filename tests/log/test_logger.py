import pytest

import torch
import numpy as np
import pandas as pd

from unittest.mock import patch

from biocircuits.log.logger import Logger


@pytest.fixture
def logger() -> Logger:
    return Logger()


def test_initial_index_is_zero(logger):
    assert logger.index == 0


def test_initial_storage_and_stage_are_empty(logger):
    assert len(logger.storage) == 0
    assert len(logger.stage) == 0


def test_df_on_just_created_logger_creates_empty_dataframe(logger):
    assert len(logger.df) == 0


def test_log_appends_entry_to_stage(logger):
    logger.log("foo", 3.0)
    assert "foo" in logger.stage
    assert len(logger.stage["foo"]) == 1
    assert logger.stage["foo"][0] == 3.0

    logger.log("foo", 5.0)
    assert len(logger.stage["foo"]) == 2
    assert logger.stage["foo"][-1] == 5.0


def test_log_dict(logger):
    logger.log({"foo": 3, "bar": 5})
    assert "foo" in logger.stage
    assert "bar" in logger.stage


def test_log_returns_self(logger):
    assert logger.log("foo", 42) is logger


def test_log_converts_0d_tensor_to_float(logger):
    logger.log("foo", torch.tensor(3.0))
    assert not torch.is_tensor(logger.stage["foo"][-1])
    assert pytest.approx(logger.stage["foo"][-1]) == 3.0


def test_log_converts_scalar_1d_tensor_to_float(logger):
    logger.log("foo", torch.tensor([3.0]))
    assert not torch.is_tensor(logger.stage["foo"][-1])
    assert pytest.approx(logger.stage["foo"][-1]) == 3.0


def test_log_converts_0d_array_to_float(logger):
    logger.log("foo", np.array(3.0))
    assert not isinstance(logger.stage["foo"][-1], np.ndarray)
    assert pytest.approx(logger.stage["foo"][-1]) == 3.0


def test_log_converts_scalar_1d_array_to_float(logger):
    logger.log("foo", np.array([3.0]))
    assert not isinstance(logger.stage["foo"][-1], np.ndarray)
    assert pytest.approx(logger.stage["foo"][-1]) == 3.0


def test_log_raises_if_input_not_convertible_to_scalar(logger):
    with pytest.raises(ValueError):
        logger.log("foo", [1.0, 2.0])

    with pytest.raises(ValueError):
        logger.log("foo", np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        logger.log("foo", torch.tensor([1.0, 2.0]))


def test_df_access_caches_result(logger):
    logger.log("foo", 5)
    logger.commit()

    with patch.object(logger, "save", wraps=logger.save) as wrapped_save:
        _ = logger.df
        wrapped_save.assert_called_once()

        # shouldn't be called again
        _ = logger.df
        wrapped_save.assert_called_once()


def test_commit_adds_stage_to_storage(logger):
    logger.log("foo", 3)
    logger.commit()

    assert len(logger.storage) == 1
    assert "foo" in logger.storage[0]
    assert logger.storage[0]["foo"] == 3


def test_commit_appends_index_to_stage_if_stage_not_empty(logger):
    index = 5
    logger.step(index)
    logger.log("foo", 3)
    logger.commit()

    assert "index" in logger.storage[0]
    assert logger.storage[0]["index"] == index


def test_empty_commit_does_not_change_storage(logger):
    logger.commit()
    assert len(logger.storage) == 0


def test_commit_averages_over_repeated_entries(logger):
    foo_list = [3, 4, 5]
    for foo in foo_list:
        logger.log("foo", foo)
    logger.commit()

    assert len(logger.storage) == 1
    assert pytest.approx(logger.storage[0]["foo"]) == sum(foo_list) / len(foo_list)


def test_commit_invalidates_df(logger):
    logger.log("foo", 5)
    logger.commit()

    # this creates df cache
    logger.save()

    # this should invalidate it
    logger.log("bar", 3.0)
    logger.commit()

    with patch.object(logger, "save", wraps=logger.save) as wrapped_save:
        _ = logger.df
        wrapped_save.assert_called_once()


def test_commit_empties_stage(logger):
    logger.log("foo", 3)
    logger.commit()

    assert len(logger.stage) == 0


def test_empty_commit_does_not_invalidate(logger):
    logger.log("foo", 5)
    logger.commit()
    logger.save()

    # should not invalidate
    logger.commit()
    with patch.object(logger, "save", wraps=logger.save) as wrapped_save:
        _ = logger.df
        wrapped_save.assert_not_called()


def test_step_commits_then_increments_index(logger):
    logger.log("bar", 5.3)
    with patch.object(logger, "commit", wraps=logger.commit) as wrapped_commit:
        logger.step()
        wrapped_commit.assert_called_once()

    assert logger.index == 1


def test_step_takes_new_index_value_as_arg(logger):
    new_index = 5
    logger.step(new_index)
    assert logger.index == new_index


def test_initialize_empties_stage(logger):
    logger.log("foo", 3)
    logger.initialize()
    assert len(logger.stage) == 0


def test_finalize_calls_step(logger):
    logger.log("foo", 5)
    with patch.object(logger, "step", wraps=logger.step) as wrapped_step:
        logger.finalize()
        wrapped_step.assert_called_once()

    assert len(logger.df) == 1


def test_str(logger):
    s = str(logger)

    assert s.startswith("Logger(")
    assert s.endswith(")")


def test_repr(logger):
    s = repr(logger)

    assert s.startswith("Logger(")
    assert s.endswith(")")


def test_stored_index_set_as_df_index(logger):
    logger.step(3)
    logger.log("foo", 5)

    logger.step(7)
    logger.log("bar", 3)

    logger.finalize()

    df = logger.df
    assert df.index.equals(pd.Index([3, 7]))
