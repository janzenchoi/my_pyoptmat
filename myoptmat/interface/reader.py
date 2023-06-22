"""
 Title:         Reader
 Description:   For loading data into PyOptMat's format
 Author:        Janzen Choi

"""
 
# Libraries
import torch, xarray
import pyoptmat.experiments as experiments

# Loads the data and reshapes it
def load_dataset(dataset:xarray.core.dataset.Dataset, device_type="cpu") -> tuple:
    n_rates = dataset.nrates
    n_samples = dataset.nsamples
    n_keep = n_rates * n_samples
    raw_data = experiments.load_results(dataset, device=torch.device(device_type))
    reshaped_data = tuple(data.reshape(data.shape[:-1] + (n_rates, n_samples))[..., :n_keep].reshape(data.shape[:-1] + (-1,)) for data in raw_data)
    return reshaped_data