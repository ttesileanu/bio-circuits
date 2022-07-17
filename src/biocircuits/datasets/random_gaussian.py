import torch

from typing import Sequence


class RandomGaussian:
    """Generate and sample from a random multivariate Gaussian.

    Attributes
    :param ndim: number of dimensions
    :param loc: mean of distribution
    :param u: orthogonal matrix, with each column vector corresponding to one axis of
        the Gaussian distribution
    :param cov: covariance matrix of the Gaussian distribution
    :param dist: `torch` distribution object
    """

    def __init__(self, scales: Sequence):
        """Generate the random Gaussian.

        :param scales: standard deviations along the different (random and orthogonal)
            directions; this is also used to infer the dimensionality of the samples
        """
        self.ndim = len(scales)
        self.loc = torch.zeros(self.ndim)

        # generate a random covariance matrix with given scales
        self.u, _ = torch.linalg.qr(torch.randn((self.ndim, self.ndim)))
        s = torch.diag(torch.FloatTensor(scales) ** 2)
        self.cov = self.u @ s @ self.u.T

        # create the distribution object that we'll sample from
        self.dist = torch.distributions.MultivariateNormal(
            loc=self.loc, covariance_matrix=self.cov
        )

    def sample(self, n: int) -> torch.Tensor:
        """Sample from the random Gaussian."""
        return self.dist.sample((n,))
