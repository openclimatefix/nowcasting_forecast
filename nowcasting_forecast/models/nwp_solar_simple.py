""" Simple model to take NWP irradence and make solar """
import logging
import os
from datetime import datetime, timezone
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


def nwp_irradence_simple_run_all_batches(
    configuration_file: Optional[str] = None, n_batches: int = 11
) -> List[ForecastSQL]:
    """Run model for all batches"""

    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow())

    # make confgiruation
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v0.yaml"
        )
    configuration = load_yaml_configuration(filename=configuration_file)

    # make dataloader
    dataloader = iter(BatchDataLoader(n_batches=n_batches, configuration=configuration))

    # loop over batch
    forecasts = []
    for i in range(n_batches):
        logger.debug(f"Running batch {i} into model")

        batch = next(dataloader)
        forecasts = forecasts + nwp_irradence_simple_run_one_batch(batch=batch)

    # select first 338 forecast
    if len(forecasts) > 338:
        forecasts = forecasts[0:338]

    # convert to sql objects
    forecasts_sql = [f.to_orm() for f in forecasts]

    # add national forecast # TODO make this from the forecast
    forecasts_sql.append(make_fake_national_forecast(t0_datetime_utc=t0_datetime_utc))

    return forecasts_sql


def nwp_irradence_simple_run_one_batch(batch: Union[dict, Batch]) -> List[Forecast]:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    # run model
    irradence_mean = nwp_irradence_simple(batch)

    # set up forecasts fields
    forecast_creation_time = datetime.now(timezone.utc)

    # TODO make input data from actual data
    input_data_last_updated = make_fake_input_data_last_updated()

    forecasts = []
    for i in range(batch.metadata.batch_size):

        # TODO make proper location
        location = Location(gsp_id=1, label="fake")

        forecast_value = ForecastValue(
            target_time=batch.metadata.t0_datetime_utc[i],
            expected_power_generation_megawatts=irradence_mean,
        )

        forecasts.append(
            Forecast(
                location=location,
                forecast_creation_time=forecast_creation_time,
                forecast_values=[forecast_value],
                input_data_last_updated=input_data_last_updated,
            )
        )

    return forecasts


def nwp_irradence_simple(batch: Batch) -> float:
    """Predictions for one batch"""
    nwp = batch.nwp

    # take solar irradence # TODO
    # print(nwp)
    # nwp.data = nwp.data.sel(channels_index=["dlwrf"])

    # take mean across all dims excpet mean
    irradence_mean = nwp.data.mean()

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 100

    return irradence_mean
