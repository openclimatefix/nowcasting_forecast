""" Simple model to take NWP irradence and make solar """
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr
from nowcasting_datamodel.fake import make_fake_input_data_last_updated
from nowcasting_datamodel.models import (
    Forecast,
    ForecastSQL,
    ForecastValueSQL,
    InputDataLastUpdatedSQL,
)
from nowcasting_datamodel.national import make_national_forecast
from nowcasting_datamodel.read import get_location, get_model
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.batch import Batch
from sqlalchemy.orm.session import Session

import nowcasting_forecast
from nowcasting_forecast import N_GSP
from nowcasting_forecast.dataloader import BatchDataLoader
from nowcasting_forecast.utils import floor_30_minutes_dt

logger = logging.getLogger(__name__)

NAME = "nwp_simple"


def nwp_irradiance_simple_run_all_batches(
    session: Session,
    configuration_file: Optional[str] = None,
    n_batches: int = 11,
    add_national_forecast: bool = True,
    n_gsps: int = N_GSP,
    batches_dir: Optional[str] = None,
) -> List[ForecastSQL]:
    """Run model for all batches"""

    logger.info("Running nwp_irradiance_simple model")

    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.now(timezone.utc))
    logger.info(f"Making forecasts for {t0_datetime_utc=}")

    # make configuration
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v0.yaml"
        )
    logger.debug(f"Loading configuration {configuration_file}")
    configuration = load_yaml_configuration(filename=configuration_file)

    if batches_dir is not None:
        configuration.output_data.filepath = Path(batches_dir)

    # make dataloader
    dataloader = iter(BatchDataLoader(n_batches=n_batches, configuration=configuration))

    input_data_last_updated = make_fake_input_data_last_updated()

    # loop over batch
    forecasts = []
    for i in range(n_batches):
        logger.debug(f"Running batch {i} into model")

        # calculate how many examples are needed
        n_examples = np.min([N_GSP - len(forecasts), configuration.process.batch_size])

        batch = next(dataloader)
        forecasts = forecasts + nwp_irradiance_simple_run_one_batch(
            batch=batch,
            n_examples=n_examples,
            session=session,
            input_data_last_updated=input_data_last_updated,
        )

    # select first 338 forecast
    if len(forecasts) > N_GSP:
        logger.debug(
            f"There are more than {N_GSP} forecasts, so just taking the first {N_GSP}. "
            "This can happen due to rounding up the examples to fit in batches"
        )
        forecasts = forecasts[0:N_GSP]

    # convert to pydantic objects, for validation
    _ = [Forecast.from_orm(f) for f in forecasts]

    if add_national_forecast:
        # add national forecast
        try:
            forecasts.append(make_national_forecast(forecasts=forecasts, n_gsps=n_gsps))
        except Exception as e:
            logger.error(e)
            # TODO remove this
            logger.error("Did not add national forecast, carry on for the moment")

    logger.info(f"Made {len(forecasts)} forecasts")

    return forecasts


def nwp_irradiance_simple_run_one_batch(
    batch: Union[dict, Batch],
    session: Session,
    n_examples: Optional[int] = None,
    input_data_last_updated: Optional[InputDataLastUpdatedSQL] = None,
) -> List[ForecastSQL]:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    if n_examples is None:
        n_examples = batch.metadata.batch_size

    # set up forecasts fields
    forecast_creation_time = datetime.now(tz=timezone.utc)

    # get model name
    model = get_model(name=NAME, version=nowcasting_forecast.__version__, session=session)

    if input_data_last_updated is None:
        # TODO make input data from actual data
        input_data_last_updated = make_fake_input_data_last_updated()

    # run model
    irradiance_mean = nwp_irradiance_simple(batch)

    forecasts = []
    for i in range(batch.metadata.batch_size):
        if i >= n_examples:
            break

        # get id from location
        gsp_id = batch.metadata.space_time_locations[i].id

        location = get_location(gsp_id=gsp_id, session=session)

        forecast_values = []
        for t_index in irradiance_mean.time_index:
            # add timezone
            target_time = (
                batch.metadata.space_time_locations[i].t0_datetime_utc.replace(tzinfo=timezone.utc)
                + timedelta(minutes=30) * t_index.values
            )

            forecast_values.append(
                ForecastValueSQL(
                    target_time=target_time,
                    expected_power_generation_megawatts=float(
                        irradiance_mean[i, t_index].values[0]
                    ),
                )
            )

        forecasts.append(
            ForecastSQL(
                model=model,
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

    # take mean across all x and y dims

    irradence_mean = nwp.data.mean(axis=(2, 3))

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 10

    return irradence_mean
