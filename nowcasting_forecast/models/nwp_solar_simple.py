""" Simple model to take NWP irradence and make solar """
import logging
import os
import xarray as xr
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union

from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.batch import Batch

import nowcasting_forecast
from nowcasting_forecast.database.fake import (
    make_fake_input_data_last_updated,
    make_fake_national_forecast,
)
from nowcasting_forecast.database.models import Forecast, ForecastSQL, ForecastValue, Location
from nowcasting_forecast.dataloader import BatchDataLoader
from nowcasting_forecast.utils import floor_30_minutes_dt

logger = logging.getLogger(__name__)


def nwp_irradiance_simple_run_all_batches(
    configuration_file: Optional[str] = None, n_batches: int = 11
) -> List[ForecastSQL]:
    """Run model for all batches"""

    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.now(timezone.utc))
    logger.info(f"Making forecasts for {t0_datetime_utc=}")

    # make confgiruation
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v0.yaml"
        )
    logger.debug(f"Loading configuration {configuration_file}")
    configuration = load_yaml_configuration(filename=configuration_file)

    # make dataloader
    dataloader = iter(BatchDataLoader(n_batches=n_batches, configuration=configuration))

    # loop over batch
    forecasts = []
    for i in range(n_batches):
        logger.debug(f"Running batch {i} into model")

        batch = next(dataloader)
        forecasts = forecasts + nwp_irradiance_simple_run_one_batch(batch=batch, batch_idx=i)

    # select first 338 forecast
    if len(forecasts) > 338:
        logger.debug(
            "There are more than 338 forecasts, so just taking the first 338. "
            "This can happen due to rounding up the examples to fit in batches"
        )
        forecasts = forecasts[0:338]

    # convert to sql objects
    forecasts_sql = [f.to_orm() for f in forecasts]

    # add national forecast # TODO make this from the forecast
    forecasts_sql.append(make_fake_national_forecast(t0_datetime_utc=t0_datetime_utc))

    logger.info(f"Made {len(forecasts_sql)} forecasts")

    return forecasts_sql


def nwp_irradiance_simple_run_one_batch(
    batch: Union[dict, Batch], batch_idx: int
) -> List[Forecast]:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    # set up forecasts fields
    forecast_creation_time = datetime.now(tz=timezone.utc)

    # TODO make input data from actual data
    input_data_last_updated = make_fake_input_data_last_updated()

    # run model
    irradiance_mean = nwp_irradiance_simple(batch)

    forecasts = []
    for i in range(batch.metadata.batch_size):

        # TODO make proper location
        gsp_id = i + batch_idx * batch.metadata.batch_size
        location = Location(gsp_id=gsp_id, label=f"GSP_{gsp_id}")

        forecast_values = []
        for t_index in irradiance_mean.time_index:
            # add timezone
            target_time = batch.metadata.t0_datetime_utc[i].replace(tzinfo=timezone.utc) \
                          + timedelta(minutes=30)

            forecast_values.append(ForecastValue(
                target_time=target_time,
                expected_power_generation_megawatts=irradiance_mean[i, t_index],
            ))

        forecasts.append(
            Forecast(
                model_name="mvp_irradence_v0",
                location=location,
                forecast_creation_time=forecast_creation_time,
                forecast_values=forecast_values,
                input_data_last_updated=input_data_last_updated,
            )
        )

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

    # take mean across all dims excpet mean
    # TODO take mean for each example
    irradence_mean = nwp.data.mean(axis=(2,3))

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 100

    return irradence_mean
