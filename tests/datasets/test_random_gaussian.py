import pytest
import torch

import numpy as np

from typing import Tuple

from biocircuits.datasets.random_gaussian import RandomGaussian


@pytest.fixture
def gaussian() -> Tuple[RandomGaussian, list]:
    torch.manual_seed(0)
    scales = [1.5, 1.2, 0.8, 0.15, 0.1]

    gauss = RandomGaussian(scales=scales)
    return gauss, scales


def test_sample_returns_correct_dimensionality(gaussian):
    n = 15
    samples = gaussian[0].sample(n)
    assert samples.shape == (n, len(gaussian[1]))


def test_sample_mean_is_close_to_zero(gaussian):
    samples = gaussian[0].sample(10_000)
    mean = torch.mean(samples, dim=0)
    assert torch.max(torch.abs(mean)) < 0.03


def test_samples_scales_match_expected(gaussian):
    n = 10_000
    samples = gaussian[0].sample(10_000)
    sample_scales = (torch.linalg.svd(samples, full_matrices=False)).S / np.sqrt(n)

    assert torch.allclose(sample_scales, torch.FloatTensor(gaussian[1]), rtol=0.03)


def test_ndim_attrib_is_correct(gaussian):
    assert gaussian[0].ndim == len(gaussian[1])


def test_loc_attrib_is_zero(gaussian):
    assert torch.allclose(gaussian[0].loc, torch.tensor(0.0))


def test_cov_attrib_matches_u_and_scales(gaussian):
    s = torch.diag(torch.FloatTensor(gaussian[1]))
    expected_cov = gaussian[0].u @ s**2 @ gaussian[0].u.T
    assert torch.allclose(gaussian[0].cov, expected_cov)


def test_cov_attrib_matches_samples(gaussian):
    samples = gaussian[0].sample(10_000)
    cov = torch.cov(samples.T)
    assert torch.allclose(cov, gaussian[0].cov, rtol=0.05, atol=0.005)
