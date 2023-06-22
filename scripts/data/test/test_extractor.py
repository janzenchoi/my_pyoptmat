import xarray as xr, torch
import warnings; warnings.filterwarnings("ignore")

def downsample(rawdata, nkeep, nrates, nsamples):
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cpu")
scale = 0.01
nsamples = 10  # at each strain rate
dataset = xr.open_dataset("~/pyoptmat/examples/structural-inference/tension/scale-%3.2f.nc" % scale)

import myoptmat.interface.converter as converter
csv_dict = converter.dataset_to_dict(dataset, 100)
converter.dict_to_csv(csv_dict, "temp.csv")