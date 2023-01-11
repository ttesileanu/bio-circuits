# %% [markdown]
# # Performing principal component analysis (PCA) using biologically plausible neural networks

# %%
import pydove as dv

import torch
import numpy as np
import tqdm.auto as tqdm

from torchvision.datasets import MNIST
import biocircuits as bio

# %% [markdown]
# ## Make a dataset -- multivariate Gaussian

# %%
torch.manual_seed(0)

scales = [1.5, 1.2, 0.8, 0.2, 0.1]
n_samples = 100_000
gaussian = bio.datasets.RandomGaussian(scales=scales)
samples = gaussian.sample(n_samples)

# %%
with dv.FigureManager() as (_, ax):
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.05)

# %% [markdown]
# ## Run similarity matching on the data

# %%
output_dim = 3
model = bio.arch.NSM(samples.shape[1], output_dim, tau=100)
checkpoint_callback = bio.log.ModelCheckpoint(frequency=10)
trainer = bio.OnlineTrainer(
    callbacks=[bio.log.ProgressBar(mininterval=0.1), checkpoint_callback]
)
output = trainer.fit(model, samples)

# %% [markdown]
## Show results

# %%
logger = trainer.logger
indices = checkpoint_callback.indices
checkpoints = checkpoint_callback.checkpoints
with dv.FigureManager(1, 2) as (fig, (ax1, ax2)):
    ax1.plot(
        indices["batch_idx"],
        [checkpoint["W"].numpy().ravel() for checkpoint in checkpoints],
        lw=0.25,
    )
    ax1.set_xlabel("sample")
    ax1.set_ylabel("weight")
    ax1.set_title("feedforward weights")

    ax2.plot(
        indices["batch_idx"],
        [checkpoint["M"].numpy().ravel() for checkpoint in checkpoints],
        lw=0.25,
    )
    ax2.set_xlabel("sample")
    ax2.set_ylabel("weight")
    ax2.set_title("lateral weights")

    fig.suptitle("Weight evolution")

# %% [markdown]
# Calculate loss function in rolling window.

# %%
window_size = 1000
window_step = 100

nsm_loss = []
top_pc_loss = []

idxs = []

a_output = np.vstack(output)
for i in tqdm.tqdm(range(0, n_samples - window_size, window_step)):
    crt_x = samples[i : i + window_size].numpy()
    crt_y = a_output[i : i + window_size]

    crt_x_cov = crt_x @ crt_x.T
    crt_y_cov = crt_y @ crt_y.T

    crt_loss = np.sum((crt_x_cov - crt_y_cov) ** 2) / window_size**2

    crt_x_proj = crt_x @ gaussian.u[:, :output_dim].numpy()
    crt_x_proj_cov = crt_x_proj @ crt_x_proj.T

    crt_proj_loss = np.sum((crt_x_proj_cov - crt_y_cov) ** 2) / window_size**2

    idxs.append(i + window_size / 2)
    nsm_loss.append(crt_loss)
    top_pc_loss.append(crt_proj_loss)

nsm_loss = np.array(nsm_loss)
top_pc_loss = np.array(top_pc_loss)

# %%
with dv.FigureManager() as (_, ax):
    ax.semilogy(idxs, nsm_loss, label="full NSM loss")
    ax.semilogy(idxs, top_pc_loss, label="top PC loss")
    ax.set_xlabel("sample")
    ax.set_ylabel("NSM loss")

    expected_residual = sum(_**4 for _ in scales[output_dim:])
    ax.axhline(
        expected_residual, c="k", lw=1.0, ls="--", label="expected loss from bottom PCs"
    )

    ax.legend(frameon=False)

# %%
