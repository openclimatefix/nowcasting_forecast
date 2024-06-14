"""Dataloader for the Power Perceiver forecaster"""
import os
from typing import Iterable, Optional

import torch
from nowcasting_dataloader.datasets import NetCDFDataset
from nowcasting_dataset.config.load import load_yaml_configuration

import nowcasting_forecast


def get_power_perceiver_dataset(
    configuration_file: Optional[str] = None,
    n_batches: int = 11,
    src_path: Optional[str] = None,
    tmp_path: Optional[str] = None,
    batch_save_dir: Optional[str] = None,
) -> Iterable[torch.utils.data.Dataset]:
    """
    Get Dataset for Power Perceiver ML model.
    """
    # make configuration
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "power_perceiver.yaml"
        )

    configuration = load_yaml_configuration(filename=configuration_file)

    if src_path is None:
        src_path = configuration.output_data.filepath / "live"
    if tmp_path is None:
        tmp_path = configuration.output_data.filepath / "live"

    data_loader = NetCDFDataset(
        n_batches=n_batches,
        src_path=src_path,
        tmp_path=tmp_path,
        configuration=configuration,
        normalize=False,  # The Helios code will normalise the data :)
        mix_two_batches=False,
        save_first_batch=os.path.join(batch_save_dir, "batchml", "batchml_0.npy")
        if batch_save_dir is not None
        else None,
    )

    return iter(data_loader)
