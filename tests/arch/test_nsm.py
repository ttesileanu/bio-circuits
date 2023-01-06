import pytest

import torch
import numpy as np

from biocircuits.arch.base import BaseOnlineModel
from biocircuits.arch.nsm import NSM


@pytest.fixture
def nsm():
    torch.manual_seed(1)
    model = NSM(5, 3)
    return model


@pytest.fixture
def data():
    return torch.FloatTensor(
        [
            [-0.5, 0.3, 0.5, 1.2, 0.1],
            [1.5, 0.0, -0.3, 0.2, 0.1],
            [0.5, -0.3, 1.3, 0.2, -0.1],
        ]
    )


def test_nsm_inherits_from_online_model(nsm):
    assert isinstance(nsm, BaseOnlineModel)


def test_initial_m_tilde_is_symmetric(nsm):
    # this is a bit trivial because right now initial m_tilde is zero!
    # m_tilde[i, j] = m[i, j] * tau[i] for i \ne j
    m_tilde = nsm.tau[:, None] * nsm.M
    assert torch.allclose(m_tilde, m_tilde.T)


def test_initial_m_has_zero_diagonal(nsm):
    # this is a bit trivial because right now the whole initial m is zero!
    assert torch.allclose(torch.diag(nsm.M), torch.tensor(0.0))


def test_initial_m_is_zero(nsm):
    assert torch.allclose(nsm.M, torch.tensor(0.0))


def test_initial_w_is_xavier():
    # need a big matrix to have good statistics
    torch.manual_seed(1)
    nsm = NSM(200, 100)

    assert torch.abs(torch.mean(nsm.W)) < 0.01
    assert torch.abs(torch.max(nsm.W) + torch.min(nsm.W)) < 0.01

    fan = sum(nsm.W.shape)
    assert torch.abs(torch.max(nsm.W) - np.sqrt(6 / fan)) < 0.01
    assert torch.abs(torch.std(nsm.W) - np.sqrt(2 / fan)) < 0.01


def test_initial_tau_is_one(nsm):
    assert torch.allclose(nsm.tau, torch.tensor(1.0))


def test_constructor_tau_float():
    tau = 3.0

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert len(nsm.tau) == 3
    assert torch.allclose(nsm.tau, torch.tensor(tau))


def test_constructor_tau_torch_scalar():
    tau = torch.tensor(3.0)

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert len(nsm.tau) == 3
    assert torch.allclose(nsm.tau, tau)


def test_constructor_tau_np_scalar():
    tau = np.array(3.0, dtype="float32")

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert len(nsm.tau) == 3
    assert torch.allclose(nsm.tau, torch.tensor(tau))


def test_constructor_tau_torch_array():
    tau = torch.FloatTensor([1.0, 3.0, 2.0])

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert torch.allclose(nsm.tau, tau)


def test_constructor_tau_np_array():
    tau = np.array([1.0, 3.0, 2.0])

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert torch.allclose(nsm.tau, torch.FloatTensor(tau))


def test_constructor_tau_list():
    tau = [1.0, 3.0, 2.0]

    torch.manual_seed(1)
    nsm = NSM(5, 3, tau=tau)

    assert torch.allclose(nsm.tau, torch.FloatTensor(tau))


def test_constructor_raises_if_initial_tau_has_wrong_size():
    with pytest.raises(ValueError):
        NSM(5, 3, tau=torch.FloatTensor([2.0, 3.0]))


def test_fast_convergence(nsm):
    nsm.fast_iterations = 1000  # use a lot of iterations to ensure convergence
    # use a non-trivial M...
    m = torch.zeros_like(nsm.M)
    torch.nn.init.xavier_uniform_(m)
    # but make sure it's symmetric and has zeros on diagonal
    m = 0.5 * (m + m.T)
    m -= torch.diag(torch.diag(m))
    nsm.M = torch.nn.Parameter(m)

    x = torch.FloatTensor([-0.1, 0.3, 0.5, 0.2, 0.4])
    y = nsm.forward(x)

    m_tilde = nsm.tau[:, None] * nsm.M + torch.diag(nsm.tau)
    w_tilde = nsm.tau[:, None] * nsm.W

    assert torch.allclose(m_tilde @ y, w_tilde @ x)


def test_one_fast_iteration():
    results = []
    n_it0 = 5

    x = torch.FloatTensor([-0.1, 0.3, 0.5, 0.2, 0.4])
    for n_it in range(n_it0, n_it0 + 2):
        torch.manual_seed(1)
        nsm = NSM(5, 3)

        nsm.fast_iterations = n_it

        # use a non-trivial M...
        m = torch.zeros_like(nsm.M)
        torch.nn.init.xavier_uniform_(m)
        # but make sure it's symmetric and has zeros on diagonal
        m = 0.5 * (m + m.T)
        m -= torch.diag(torch.diag(m))
        nsm.M = torch.nn.Parameter(m)

        results.append(nsm.forward(x))

    y_exp = nsm.W @ x - nsm.M @ results[0]

    assert torch.allclose(results[1], y_exp)


def test_fast_iteration_with_batch():
    seed = 100

    x = torch.FloatTensor([[-0.1, 0.3, 0.5, 0.2, 0.4], [0.2, 0.5, 0.3, 0.1, -0.5]])

    torch.manual_seed(seed)
    nsm = NSM(5, 3)

    y_batch = nsm.forward(x)

    y = []
    for crt_x in x:
        torch.manual_seed(seed)
        nsm = NSM(5, 3)

        y.append(nsm.forward(crt_x))

    assert torch.allclose(y_batch, torch.stack(y))


def test_grad_tau(nsm):
    x = torch.FloatTensor([-0.5, 0.3, 0.5, 1.2, 0.1])
    y = nsm.forward(x)
    nsm.backward(x, y)
    assert torch.allclose(nsm.tau.grad, -(y**2))


def test_grad_w(nsm):
    x = torch.FloatTensor([-0.5, 0.3, 0.5, 1.2, 0.1])
    y = nsm.forward(x)
    nsm.backward(x, y)

    new_tau = nsm.tau + y**2
    for i in range(nsm.output_dim):
        for j in range(nsm.input_dim):
            expected = y[i] * (nsm.W[i, j] * y[i] - x[j]) / new_tau[i]
            assert torch.allclose(nsm.W.grad[i, j], expected)


def test_grad_m(nsm):
    x = torch.FloatTensor([-0.5, 0.3, 0.5, 1.2, 0.1])
    y = nsm.forward(x)
    nsm.backward(x, y)

    new_tau = nsm.tau + y**2
    for i in range(nsm.output_dim):
        for j in range(nsm.output_dim):
            if i == j:
                expected = torch.tensor(0.0)
            else:
                expected = y[i] * (nsm.M[i, j] * y[i] - y[j]) / new_tau[i]
            assert torch.allclose(nsm.M.grad[i, j], expected)


@pytest.mark.parametrize("var", ["tau", "M", "W"])
def test_batch_grad(nsm, var, data):
    x = data
    y = nsm.forward(x)
    nsm.backward(x, y)

    initial = getattr(nsm, var).detach().clone()
    batch_grad = getattr(nsm, var).grad.detach().clone()

    optimizer = torch.optim.SGD(nsm.parameters(), lr=1.0)

    for crt_x, crt_y in zip(x, y):
        optimizer.zero_grad()
        nsm.backward(crt_x, crt_y)
        optimizer.step()

    expected_change = getattr(nsm, var) - initial
    assert torch.allclose(expected_change, -batch_grad)


@pytest.mark.parametrize("var", ["tau", "M", "W"])
def test_backward_accumulates_gradients(nsm, var):
    x = torch.FloatTensor([[-0.5, 0.3, 0.5, 1.2, 0.1], [1.5, 0.0, -0.3, 0.2, 0.1]])

    indep_grads = []
    for crt_x in x:
        # reset the gradient
        getattr(nsm, var).grad = None

        crt_y = nsm.forward(crt_x)
        nsm.backward(crt_x, crt_y)
        indep_grads.append(getattr(nsm, var).grad)

    # now do the same but don't reset every time
    getattr(nsm, var).grad = None
    for crt_x in x:
        crt_y = nsm.forward(crt_x)
        nsm.backward(crt_x, crt_y)

    total_grad = getattr(nsm, var).grad
    expected = sum(indep_grads)

    assert torch.allclose(total_grad, expected)


def test_ensure_forward_does_not_track_gradients(nsm):
    x = torch.FloatTensor([1.0, -0.1, 2.0, 0.5, 0.3]).requires_grad_()
    y = nsm.forward(x)

    assert not y.requires_grad


def test_configure_optimizer_returns_an_sgd_optimizer_and_no_scheduler(nsm):
    optimizers, schedulers = nsm.configure_optimizers()

    assert len(optimizers) == 1
    assert len(schedulers) == 0

    assert isinstance(optimizers[0], torch.optim.SGD)


def test_configure_optimizer_default_lr_is_one(nsm):
    optimizers, _ = nsm.configure_optimizers()
    assert pytest.approx(optimizers[0].param_groups[0]["lr"]) == 1.0


def test_configure_optimizer_one_param_group(nsm):
    optimizers, _ = nsm.configure_optimizers()
    assert len(optimizers[0].param_groups) == 1


def test_configure_optimizer_lr(nsm):
    with pytest.raises(ValueError):
        nsm.configure_optimizers(lr=0.05)


def test_configure_optimizer_other_kwargs(nsm):
    momentum = 0.05
    optimizers, _ = nsm.configure_optimizers(momentum=momentum)
    assert pytest.approx(optimizers[0].param_groups[0]["momentum"]) == momentum


def test_w_tilde_trains_to_sum_of_y_x(nsm, data):
    nsm.W = torch.nn.Parameter(torch.zeros_like(nsm.W))

    optimizers, _ = nsm.configure_optimizers()
    optimizer = optimizers[0]

    out = data[:, : nsm.output_dim]
    for x, y in zip(data, out):
        optimizer.zero_grad()
        nsm.backward(x, y)
        optimizer.step()

    w_tilde = nsm.tau[:, None] * nsm.W
    expected_w_tilde = sum(out[:, :, None] @ data[:, None, :])

    assert torch.allclose(w_tilde, expected_w_tilde)


def test_m_tilde_trains_to_sum_of_y_y(nsm, data):
    nsm.tau = torch.nn.Parameter(torch.zeros_like(nsm.tau))
    nsm.M = torch.nn.Parameter(torch.zeros_like(nsm.M))

    optimizers, _ = nsm.configure_optimizers()
    optimizer = optimizers[0]

    out = 0.3 * data[:, : nsm.output_dim] - 0.5
    for x, y in zip(data, out):
        optimizer.zero_grad()
        nsm.backward(x, y)
        optimizer.step()

    m_tilde = nsm.tau[:, None] * nsm.M + torch.diag(nsm.tau)
    expected_m_tilde = sum(out[:, :, None] @ out[:, None, :])

    assert torch.allclose(m_tilde, expected_m_tilde)


def test_tau_trains_to_sum_of_y_squared(nsm, data):
    nsm.tau = torch.nn.Parameter(torch.zeros_like(nsm.tau))

    optimizers, _ = nsm.configure_optimizers()
    optimizer = optimizers[0]

    out = -0.5 * data[:, : nsm.output_dim] + 0.1
    for x, y in zip(data, out):
        optimizer.zero_grad()
        nsm.backward(x, y)
        optimizer.step()

    expected_tau = sum(out**2)

    assert torch.allclose(nsm.tau, expected_tau)


def test_one_fast_iteration_relu():
    results = []
    n_it0 = 5

    x = torch.FloatTensor([-0.1, 0.3, 0.5, 0.2, 0.4])
    for n_it in range(n_it0, n_it0 + 2):
        torch.manual_seed(1)
        nsm = NSM(5, 3, activation="relu")

        nsm.fast_iterations = n_it

        # use a non-trivial M...
        m = torch.zeros_like(nsm.M)
        torch.nn.init.xavier_uniform_(m)
        # but make sure it's symmetric and has zeros on diagonal
        m = 0.5 * (m + m.T)
        m -= torch.diag(torch.diag(m))
        nsm.M = torch.nn.Parameter(m)

        results.append(nsm.forward(x))

    y_exp = (nsm.W @ x - nsm.M @ results[0]).clip_(min=0)

    assert torch.allclose(results[1], y_exp)


def test_test_step_returns_output_of_forward_call(nsm, data):
    output = nsm.test_step(data)
    expected = nsm(data)

    assert torch.allclose(output, expected)


def test_training_step_returns_output_of_forward_call_before_update(nsm, data):
    nsm.configure_optimizers()

    expected = nsm(data)
    output = nsm.training_step(data)

    assert torch.allclose(output, expected)


def test_training_step_updates_weights(nsm, data):
    nsm.configure_optimizers()

    W0 = nsm.W.detach().clone()
    M0 = nsm.M.detach().clone()
    nsm.training_step(data)

    assert torch.max(torch.abs(W0 - nsm.W)) > 1e-3
    assert torch.max(torch.abs(M0 - nsm.M)) > 1e-3
