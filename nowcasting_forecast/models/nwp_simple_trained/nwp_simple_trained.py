""" Run the nwp simple trained model

1. Setup model and load weights from s3/neptune
2. run model by looping over batches
3. format predictions
4. add national forecast
"""


import logging
import os
from datetime import timedelta, timezone
from typing import Optional, Union

import pandas as pd
import xarray as xr
from nowcasting_dataset.dataset.batch import Batch

import nowcasting_forecast
from nowcasting_forecast.models.nwp_simple_trained.xr_utils import re_order_dims

logger = logging.getLogger(__name__)

NAME = "nwp_simple_trained"


def nwp_irradiance_simple_trained_run_one_batch(
    batch: Union[dict, Batch],
    pytorch_model,
    n_examples: Optional[int] = None,
) -> pd.DataFrame:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    if n_examples is None:
        n_examples = batch.metadata.batch_size

    # run model
    predictions = nwp_irradiance_simple_trained(batch, model=pytorch_model)

    forecasts = []
    for i in range(batch.metadata.batch_size):
        if i >= n_examples:
            break

        # get id from location
        gsp_id = batch.metadata.space_time_locations[i].id

        # t0 value value
        t0_datetime_utc = batch.metadata.space_time_locations[i].t0_datetime_utc.replace(
            tzinfo=timezone.utc
        )

        for t_index in range(len(predictions[0])):
            # add timezone
            target_time = t0_datetime_utc + timedelta(minutes=30) * (t_index + 1)

            forecasts.append(
                dict(
                    t0_datetime_utc=t0_datetime_utc,
                    target_datetime_utc=target_time,
                    forecast_gsp_pv_outturn_mw=float(predictions[i, t_index]),
                    gsp_id=gsp_id,
                )
            )

    forecasts = pd.DataFrame(forecasts)

    return forecasts


def nwp_irradiance_simple_trained(batch: Batch, model) -> xr.DataArray:
    """Predictions for one batch

    Just to take the mean NWP data across the batch

    Returns a data array of nwp irradiance with dimensions for examples and time.
    """

    nwp = batch.nwp

    nwp = re_order_dims(nwp)

    # move to torch
    nwp_batch_ml = nwp.torch.to_tensor(["data", "time", "init_time", "x_osgb", "y_osgb"])

    nwp = nwp_batch_ml["data"]

    # normalize
    dswrf_mean = 294.6696933986283
    dswrf_std = 233.1834250473355
    nwp = nwp - dswrf_mean
    nwp = nwp / dswrf_std

    # run model
    predictions = model(nwp)

    # re-normalize
    # load capacity
    capacity = pd.read_csv(
        os.path.join(os.path.dirname(nowcasting_forecast.__file__), "data", "gsp_capacity.csv"),
        index_col=["gsp_id"],
    )
    capacity = capacity.loc[batch.metadata.ids]

    # multiply predictions by capacities
    predictions = capacity.values * predictions.detach().cpu().numpy()

    return predictions
