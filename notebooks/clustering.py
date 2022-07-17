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
dataset = torch.utils.data.TensorDataset(samples)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

output_dim = n_clusters
model = bio.arch.NSM(samples.shape[1], output_dim, tau=5e4, activation="relu")
trainer = bio.OnlineTrainer(model, dataloader)
for batch in tqdm.tqdm(trainer):
    batch.training_step()
    if batch.every(10):
        batch.log("w", model.W)
        batch.log("m", model.M)

# %% [markdown]
## Show results

# %%
with dv.FigureManager(1, 2) as (fig, (ax1, ax2)):
    ax1.plot(
        trainer.log["w.sample"],
        np.reshape(trainer.log["w"], (-1, model.input_dim * model.output_dim)),
        lw=0.25,
    )
    ax1.set_xlabel("sample")
    ax1.set_ylabel("weight")
    ax1.set_title("feedforward weights")

    ax2.plot(
        trainer.log["m.sample"],
        np.reshape(trainer.log["m"], (-1, model.output_dim * model.output_dim)),
        lw=0.25,
    )
    ax2.set_xlabel("sample")
    ax2.set_ylabel("weight")
    ax2.set_title("lateral weights")

    fig.suptitle("Weight evolution")

# %%
cluster_assignments = np.argmax(trainer.log["output"], axis=1)
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
