"""Dataloader for the CNN forecaster"""
import os
from typing import Optional

from nowcasting_dataloader.datasets import NetCDFDataset
from nowcasting_dataset.config.load import load_yaml_configuration

import nowcasting_forecast


def get_cnn_data_loader(
    configuration_file: Optional[str] = None,
    n_batches: Optional[int] = 11,
    src_path: Optional[str] = None,
    tmp_path: Optional[str] = None,
):
    """
    Get data laoder for cnn model

    configuration_file:
    """
    # make configuration
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v2.yaml"
        )

    configuration = load_yaml_configuration(filename=configuration_file)

    if src_path is None:
        src_path = configuration.output_data.filepath
    if tmp_path is None:
        tmp_path = configuration.output_data.filepath

    data_loader = NetCDFDataset(
        n_batches=n_batches, src_path=os.path.join(src_path, "live"), tmp_path=os.path.join(tmp_path, "live"), configuration=configuration
    )

    return iter(data_loader)
