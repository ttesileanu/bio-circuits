import pytest

import torch
import numpy as np

from biocircuits.arch.nsm import NSM


@pytest.fixture
def nsm():
    torch.manual_seed(1)
    model = NSM(5, 3)
    return model


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
