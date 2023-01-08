import pytest

import torch
import numpy as np

from biocircuits.log.logger import Logger


@pytest.fixture
def logger() -> Logger:
    return Logger()


def test_history_is_empty_if_nothing_is_added(logger):
    logger.finalize()
    assert len(logger.history) == 0


def test_log_adds_history_entry(logger):
    logger.log("foo", torch.tensor(3.0))
    assert "foo" in logger.history


def test_tensors_coalesced_after_finalize(logger):
    a = 3.0
    b = 2.0
    logger.log("foo", torch.tensor(a))
    logger.log("foo", torch.tensor(b))
    logger.finalize()

    assert "foo" in logger.history
    np.testing.assert_allclose(logger.history["foo"], [a, b])


def test_chaining_log_calls(logger):
    logger.log("foo", torch.tensor(2.0)).log("foo", torch.tensor(3.0))
    logger.finalize()

    assert len(logger.history["foo"]) == 2


def test_finalized_history_contains_arrays(logger):
    a = 3.0
    b = 2.0
    logger.log("foo", torch.tensor(a))
    logger.log("foo", torch.tensor(b))
    logger.finalize()

    assert isinstance(logger.history["foo"], np.ndarray)


def test_access_history_directly_using_square_brackets(logger):
    logger.log("foo", torch.tensor(3.0))
    logger.finalize()
    np.testing.assert_allclose(logger["foo"], 3.0)


def test_storing_arrays(logger):
    a = 3.0
    b = 2.0
    logger.log("foo", np.array(a))
    logger.log("foo", np.array(b))
    logger.finalize()

    np.testing.assert_allclose(logger["foo"], [a, b])


def test_log_scalar_nontensors_or_arrays(logger):
    x = 3.0
    logger.log("bar", x)
    logger.finalize()

    np.testing.assert_allclose(logger["bar"], [x])


def test_access_raises_if_called_for_inexistent_field(logger):
    logger.finalize()
    with pytest.raises(KeyError):
        logger["foo"]


def test_log_ints_leads_to_int64_array(logger):
    i = 3
    logger.log("bar", i)
    logger.finalize()

    assert logger["bar"].dtype == "int64"


def test_log_list_makes_perlayer_entries(logger):
    x = [torch.FloatTensor([1.0]), torch.FloatTensor([2.0, 3.0])]
    logger.log("x", x)
    logger.finalize()

    for i in range(len(x)):
        assert f"x:{i}" in logger.history
        np.testing.assert_allclose(logger[f"x:{i}"][-1], x[i])


def test_log_adds_row_index_to_tensors(logger):
    x = torch.FloatTensor([1.0, 3.0])
    logger.log("x", x)
    logger.finalize()

    assert logger["x"].shape == (1, len(x))


def test_log_stacks_tensors_properly(logger):
    x = torch.FloatTensor([[1.0, 3.0], [4.0, 5.0]])
    for row in x:
        logger.log("x", row)
    logger.finalize()

    np.testing.assert_allclose(logger["x"], x)


def test_str(logger):
    logger.log("x", 0)
    s = str(logger)

    assert s.startswith("Logger(")
    assert s.endswith(")")
    assert "x" in s


def test_repr(logger):
    logger.log("x", 0)
    logger.log("bar", 0)
    s = repr(logger)

    assert s.startswith("Logger(")
    assert s.endswith(")")
    assert "x" in s
    assert "bar" in s


def test_log_works_with_tensor_that_requires_grad(logger):
    x = torch.FloatTensor([1.0, 2.0]).requires_grad_()
    logger.log("x", x)
    logger.finalize()

    assert isinstance(logger["x"], np.ndarray)


def test_log_clones_tensor(logger):
    y = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logger.log("y", y)

    # change tensor after logging, see if change persists
    y_orig = y.detach().clone()
    y[0, 1] = -2.0
    logger.finalize()

    np.testing.assert_allclose(logger["y"][-1], y_orig)


def test_log_higher_dim_tensor(logger):
    x = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.3], [0.5, 0.3, 0.4]])
    logger.log("foo", x)
    logger.log("foo", y)
    logger.finalize()

    assert len(logger["foo"]) == 2
    np.testing.assert_allclose(logger["foo"][0], x)
    np.testing.assert_allclose(logger["foo"][1], y)


def test_log_batch(logger):
    logger.log_batch("foo", torch.FloatTensor([1, 2, 3]))
    logger.log_batch("bar", torch.FloatTensor([[1, 2], [3, 4], [5, 6]]))
    logger.finalize()

    for var in ["foo", "bar"]:
        assert len(logger[var]) == 3


def test_log_multiple_fields_at_once(logger):
    x = torch.FloatTensor([1, 2, 3])
    y = torch.FloatTensor([[2, 3, 4], [-1, 0.5, 2]])
    logger.log({"x": x, "y": y})
    logger.finalize()

    assert "x" in logger.history
    assert "y" in logger.history

    assert len(logger["x"]) == 1
    assert len(logger["y"]) == 1

    np.testing.assert_allclose(logger["x"][0], x)
    np.testing.assert_allclose(logger["y"][0], y)


def test_log_batch_multiple_fields(logger):
    x = torch.FloatTensor([0.5, 1, 1.5])
    y = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    logger.log_batch({"x": x, "y": y})
    logger.finalize()

    assert logger["x"].shape == x.shape
    assert logger["y"].shape == y.shape

    np.testing.assert_allclose(logger["x"], x)
    np.testing.assert_allclose(logger["y"], y)


def test_log_batch_multilayer(logger):
    w0 = torch.FloatTensor([[1, 2.5], [3, 2.2]])
    w1 = torch.FloatTensor([[-0.1, 1.5, 0.5], [0.2, 2.3, -1.2]])
    logger.log_batch("w", [w0, w1])
    logger.finalize()

    assert "w:0" in logger.history
    assert "w:1" in logger.history
    assert logger["w:0"].shape == w0.shape
    assert logger["w:1"].shape == w1.shape


def test_accumulate_followed_by_log_accumulated_averages(logger):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for crt_x in x:
        logger.accumulate("x", crt_x)
    logger.log_accumulated()
    logger.finalize()

    assert len(logger["x"]) == 1
    np.testing.assert_allclose(logger["x"][-1], x.mean(dim=0))


def test_log_accumulated_resets_accumulator(logger):
    xs = [1.5, 2.3]
    logger.accumulate("x", xs[0])
    logger.log_accumulated()

    logger.accumulate("x", xs[1])
    logger.log_accumulated()

    logger.finalize()

    assert pytest.approx(logger["x"][-1]) == xs[1]


def test_log_iterable_of_non_tensors(logger):
    logger.log("x", [0, 0])
    logger.finalize()

    assert len(logger["x:0"]) == 1
    assert len(logger["x:1"]) == 1


def test_log_dict_with_non_tensors(logger):
    logger.log({"x": torch.tensor(1.0), "y": 2.0})
    logger.finalize()

    assert len(logger["x"]) == 1
    assert len(logger["y"]) == 1


def test_calculate_accumulated(logger):
    values = [5.0, 2.5, -0.3]
    for value in values:
        logger.accumulate("x", value)
    mean = logger.calculate_accumulated("x")

    assert pytest.approx(mean) == np.mean(values)


def test_calculate_accumulated_has_the_right_dimensions(logger):
    values = [np.array([5.0, -0.3]), np.array([2.5, 0.7])]
    for value in values:
        logger.accumulate("x", value)
    mean = logger.calculate_accumulated("x")

    assert mean.shape == values[0].shape


def test_calculate_accumulated_does_not_clear_accumulator(logger):
    values = [0.5, -0.3]
    logger.accumulate("x", values[0])
    assert pytest.approx(logger.calculate_accumulated("x").item()) == values[0]

    logger.accumulate("x", values[1])
    accum_val = logger.calculate_accumulated("x")
    assert pytest.approx(accum_val) != values[1]
    assert pytest.approx(accum_val) == np.mean(values)


def test_calculate_accumulated_returns_nan_if_empty_field(logger):
    a = logger.calculate_accumulated("x")
    assert np.isnan(a)


def test_logging_scalars_leads_to_vector_history_float(logger):
    logger.log("x", 0.0)
    logger.log("x", 0.5)
    logger.finalize()

    assert logger["x"].ndim == 1


def test_logging_scalars_leads_to_vector_history_int(logger):
    logger.log("x", 0)
    logger.log("x", 5)
    logger.finalize()

    assert logger["x"].ndim == 1


def test_accumulate_dict(logger):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for crt_x in x:
        logger.accumulate({"x": crt_x, "y": -crt_x})
    logger.log_accumulated()
    logger.finalize()

    assert "x" in logger.history
    assert "y" in logger.history
