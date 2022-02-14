""" Useful functions for xarray objects

1. xr array and xr dataset --> to torch functions
"""
from typing import List, Optional

import torch
import xarray as xr


def register_xr_data_array_to_tensor():
    """Add torch object to data array"""
    if not hasattr(xr.DataArray, "torch"):

        @xr.register_dataarray_accessor("torch")
        class TorchAccessor:
            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            def to_tensor(self):
                """Convert this DataArray to a torch.Tensor"""
                return torch.tensor(self._obj.data, dtype=torch.float32)

            # torch tensor names does not working in dataloader yet - 2021-10-15
            # https://discuss.pytorch.org/t/collating-named-tensors/78650
            # https://github.com/openclimatefix/nowcasting_dataset/issues/25
            # def to_named_tensor(self):
            #     """Convert this DataArray to a torch.Tensor with named dimensions"""
            #     import torch
            #
            #     return torch.tensor(self._obj.data, names=self._obj.dims)


def register_xr_data_set_to_tensor():
    """Add torch object to dataset"""
    if not hasattr(xr.Dataset, "torch"):

        @xr.register_dataset_accessor("torch")
        class TorchAccessor:
            def __init__(self, xdataset_obj: xr.Dataset):
                self._obj = xdataset_obj

            def to_tensor(self, data_vars: List[str]) -> dict:
                """Convert this Dataset to dictionary of torch tensors"""
                torch_dict = {}

                for data_var in data_vars:
                    v = getattr(self._obj, data_var)

                    if data_var.find("time") != -1:
                        time_int = v.data.astype(int)
                        torch_dict[data_var] = torch.tensor(time_int, dtype=torch.float64)

                    else:
                        torch_dict[data_var] = v.torch.to_tensor()

                return torch_dict


register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


def re_order_dims(xr_dataset: xr.Dataset, expected_dims_order: Optional = None):
    """
    Re order dims to B,C,T,H,W
    """
    if expected_dims_order is None:
        expected_dims_order = (
            "example",
            "channels_index",
            "time_index",
            "y_osgb_index",
            "x_osgb_index",
        )

    if xr_dataset.data.dims != expected_dims_order:
        xr_dataset.__setitem__("data", xr_dataset.data.transpose(*expected_dims_order))

    return xr_dataset
