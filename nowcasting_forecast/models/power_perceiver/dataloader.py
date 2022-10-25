"""Power Perceiver DataPipes based data loader"""
import logging
import os
from typing import Optional

from ocf_datapipes.production.power_perceiver import power_perceiver_production_datapipe
from torch.utils.data.dataloader import DataLoader

import nowcasting_forecast

logger = logging.getLogger(__name__)


def get_power_perceiver_data_loader(
    configuration_file: Optional[str] = None,
    n_batches: Optional[int] = 11,
    src_path: Optional[str] = None,
):
    """Get the Power Perceiver production Datapipes data loader"""
    logger.debug("Making Power Perceiver Data Loader")

    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "pp_v1.yaml"
        )
    datapipes = power_perceiver_production_datapipe(configuration_file)

    logger.debug("Done making Power Perceiver Data Loader")
    data_loader = DataLoader(dataset=datapipes, batch_size=None)
    return iter(data_loader)
