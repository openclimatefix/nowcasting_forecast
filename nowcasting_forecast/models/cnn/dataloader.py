"""Dataloader for the CNN forecaster"""
import logging
import os
from typing import Optional

from nowcasting_dataloader.datasets import NetCDFDataset
from nowcasting_dataset.config.load import load_yaml_configuration

import nowcasting_forecast

logger = logging.getLogger(__name__)


def get_cnn_data_loader(
    configuration_file: Optional[str] = None,
    n_batches: Optional[int] = 11,
    src_path: Optional[str] = None,
    tmp_path: Optional[str] = None,
    batch_save_dir: Optional[str] = None,
):
    """
    Get data laoder for cnn model

    configuration_file:
    """
    logger.debug("Making CNN data loader")

    # make configuration
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "cnn_v1.yaml"
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
        mix_two_batches=False,
        save_first_batch=os.path.join(batch_save_dir, "batchml", "batchml_0.npy")
        if batch_save_dir is not None
        else None,
    )

    logger.debug("Done making CNN data loader.")

    return iter(data_loader)
