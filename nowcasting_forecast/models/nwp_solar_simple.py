""" Simple model to take NWP irradence and make solar """
import logging
from datetime import timedelta, timezone
from typing import Optional, Union

import pandas as pd
import xarray as xr
from nowcasting_dataset.dataset.batch import Batch

logger = logging.getLogger(__name__)

NAME = "nwp_simple"


def nwp_irradiance_simple_run_one_batch(
    batch: Union[dict, Batch],
    n_examples: Optional[int] = None,
) -> pd.DataFrame:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    if n_examples is None:
        n_examples = batch.metadata.batch_size

    # run model
    irradiance_mean = nwp_irradiance_simple(batch)

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

        for t_index in irradiance_mean.time_index:
            target_time = t0_datetime_utc + timedelta(minutes=30) * t_index.values

            forecasts.append(
                dict(
                    t0_datetime_utc=t0_datetime_utc,
                    target_datetime_utc=target_time,
                    forecast_gsp_pv_outturn_mw=float(irradiance_mean[i, t_index].values[0]),
                    gsp_id=gsp_id,
                )
            )

    forecasts = pd.DataFrame(forecasts)

    return forecasts


def nwp_irradiance_simple(batch: Batch) -> xr.DataArray:
    """Predictions for one batch

    Just to take the mean NWP data across the batch

    Returns a data array of nwp irradiance with dimensions for examples and time.
    """
    nwp = batch.nwp

    # take solar irradence # TODO
    # print(nwp)
    # nwp.data = nwp.data.sel(channels_index=["dlwrf"])

    # take mean across all x and y dims

    irradence_mean = nwp.data.mean(axis=(2, 3))

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 10

    return irradence_mean
