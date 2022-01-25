""" Dataset and functions"""
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from nowcasting_dataset.config.model import Configuration

# from nowcasting_dataset.consts import (
#     DEFAULT_REQUIRED_KEYS,
#     GSP_DATETIME_INDEX,
#     GSP_ID,
#     GSP_YIELD,
#     NWP_DATA,
#     PV_SYSTEM_ID,
#     PV_YIELD,
#     SATELLITE_DATA,
#     TOPOGRAPHIC_DATA,
# )
from nowcasting_dataset.dataset.batch import Batch, Example

logger = logging.getLogger(__name__)


class BatchDataLoader:
    """
    Loads batches
    """

    def __init__(
        self,
        n_batches: int,
        configuration: Configuration,
    ):
        """
        Netcdf Dataset

        Args:
            n_batches: Number of batches available on disk.
            configuration: configuration object
        """
        self.n_batches = n_batches
        self.configuration = configuration

        self.src_path = os.path.join(configuration.output_data.filepath, "live")

        logger.info(f"Setting up BatchDataLoader for {self.src_path}")

    def __len__(self):
        """Length of dataset"""
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> dict:
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        logger.debug(f"Getting batch {batch_idx}")
        if not 0 <= batch_idx < self.n_batches:
            raise IndexError(
                "batch_idx must be in the range" f" [0, {self.n_batches}), not {batch_idx}!"
            )

        batch: Batch = Batch.load_netcdf(
            self.src_path,
            batch_idx=batch_idx,
            data_sources_names=["nwp"],
        )

        batch: dict = batch.dict()

        return batch
