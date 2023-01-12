# %% [markdown]
# # Clustering using biologically plausible neural networks

# %%
import pydove as dv

import torch
import numpy as np
import tqdm.auto as tqdm

from torchvision.datasets import MNIST
import biocircuits as bio

# %% [markdown]
# ## Make a dataset -- a mixture of Gaussians

# %%
torch.manual_seed(0)

input_dim = 50

n_clusters = 5
n_samples_each = 10_000
samples0 = []

sample_ids = []

for i in range(n_clusters):
    mu = 2.5 * torch.randn(input_dim)
    crt_gaussian = bio.datasets.RandomGaussian(scales=input_dim * [1.0])
    crt_samples = mu + crt_gaussian.sample(n_samples_each)

    samples0.append(crt_samples)
    sample_ids.extend(n_samples_each * [i])

n_samples = n_clusters * n_samples_each

sample_ids = torch.LongTensor(sample_ids)

perm = torch.randperm(n_samples)
samples = torch.vstack((samples0))[perm]
sample_ids = sample_ids[perm]

# %%
with dv.FigureManager() as (_, ax):
    ax.scatter(
        samples[:, 0].numpy(),
        samples[:, 1].numpy(),
        s=2,
        c=sample_ids.numpy(),
        alpha=0.1,
    )
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")

# %% [markdown]
# ## Run NSM on the data

# %%

output_dim = n_clusters
model = bio.arch.NSM(samples.shape[1], output_dim, tau=5e4, activation="relu")
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
# %%
a_output = np.vstack(output)
cluster_assignments = np.argmax(a_output, axis=1)
with dv.FigureManager() as (_, ax):
    ax.scatter(
        samples[:, 0].numpy(),
        samples[:, 1].numpy(),
        s=2,
        c=cluster_assignments,
        alpha=0.1,
    )
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.set_title("Colored by inferred clusters")

# %%
confusion = np.zeros((n_clusters, n_clusters))
for i in range(n_clusters):
    selection_gt = sample_ids.numpy() == i
    n_gt = np.sum(selection_gt)
    for j in range(n_clusters):
        selection_assigned = cluster_assignments == j

        confusion[i, j] = np.sum(selection_gt & selection_assigned) / n_gt

with dv.FigureManager() as (_, ax):
    h = ax.imshow(confusion)
    ax.set_xlabel("inferred")
    ax.set_ylabel("ground truth")

    dv.colorbar(h)

# %%
