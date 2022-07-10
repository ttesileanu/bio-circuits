import numpy as np
import torch
from torch import nn

from typing import Union, Sequence


class NSM(nn.Module):
    """Implementation of the biologically plausible online similarity matching
    algorithm.

    Despite the name, this implementation allows for both unconstrained and non-negative
    similarity matching (see `activation`).

    This follows the algorithm from [Hu, Pehlevan, Chklovskii, 2014](https://ieeexplore.ieee.org/abstract/document/7226488).

    The feed-forward weights are initialized using Xavier initialization and the later
    weights are initialized to zero.

    Attributes
    :param input_dim: dimension of input samples
    :param output_dim: dimension of output samples
    :param W: feed-forward connection weights
    :param M: lateral connection weights
    :param tau: sum of squared outputs (called $Y$ in the paper)
    :param fast_iterations: number of fast-dynamics iterations
    :param activation: activation function; can be `"linear"` or `"relu"`
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tau: Union[float, np.ndarray, torch.Tensor, Sequence] = 1.0,
        fast_iterations: int = 100,
    ):
        """Initialize the similarity matching circuit.

        :param input_dim: dimension of input samples
        :param output_dim: dimension of output samples
        :param tau: initial value(s) for the sum of squared outputs
        :param fast_iterations: number of fast-dynamics iterations
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fast_iterations = fast_iterations

        self.W = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        self.M = nn.Parameter(torch.zeros(self.output_dim, self.output_dim))

        if torch.is_tensor(tau):
            tau = tau.detach()
            if tau.ndim == 0 or len(tau) == 1:
                tau = torch.tile(tau, (self.output_dim,))
            else:
                tau = tau.clone()
        else:
            if np.size(tau) == 1:
                tau = np.repeat(tau, self.output_dim)
            else:
                tau = np.asarray(tau)
            tau = torch.FloatTensor(tau)

        if len(tau) != self.output_dim:
            raise ValueError("wrong shape for initial tau")
        self.tau = nn.Parameter(tau)

        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fast (neural) dynamics and return the neural activations.

        NB: this updates all the components of the output `y` using the old values of
        `y`, i.e.,

            y_new = W @ x + M @ y_old

        This is different from the standard implementation from (Hu et al., 2014), where
        the coordinates are updated in order, so that, e.g., `y_new[2]` is updated using
        the *new* value of the first component, `y_new[0]`, as opposed to the old value,
        `y_old[0]`.

        The initial value of `y` in the iteration is simply given by the feed-forward
        result,

            y_init = W @ x
        """
        Wx = self.W @ x

        y = Wx
        for _ in range(self.fast_iterations):
            y = self.W @ x - self.M @ y

        return y
