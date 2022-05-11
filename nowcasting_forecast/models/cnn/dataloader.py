from typing import Optional
import os

from nowcasting_dataloader.datasets import NetCDFDataset
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.config.load import load_yaml_configuration
import nowcasting_forecast


def get_cnn_data_loader(configuration_file: Optional[str] = None, n_batches: Optional[int] = 11):
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

    src_path = configuration.output_data.filepath
    tmp_path = configuration.output_data.filepath
    data_loader = NetCDFDataset(
        n_batches=n_batches, src_path=src_path, tmp_path=tmp_path, configuration=configuration
    )

    return data_loader
