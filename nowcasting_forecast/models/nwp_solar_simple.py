""" Simple model to take NWP irradence and make solar """
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

import numpy as np
import xarray as xr
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from sqlalchemy.orm.session import Session

import nowcasting_forecast
from nowcasting_forecast import N_GSP
from nowcasting_forecast.database.fake import make_fake_input_data_last_updated
from nowcasting_forecast.database.models import Forecast, ForecastSQL, ForecastValue, Location
from nowcasting_forecast.database.national import make_national_forecast
from nowcasting_forecast.database.read import get_location
from nowcasting_forecast.dataloader import BatchDataLoader
from nowcasting_forecast.utils import floor_30_minutes_dt

logger = logging.getLogger(__name__)


def nwp_irradiance_simple_run_all_batches(
    session: Session,
    configuration_file: Optional[str] = None,
    n_batches: int = 11,
    add_national_forecast: bool = True,
    n_gsps: int = N_GSP,
) -> List[ForecastSQL]:
    """Run model for all batches"""

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

    # make dataloader
    dataloader = iter(BatchDataLoader(n_batches=n_batches, configuration=configuration))

    # loop over batch
    forecasts = []
    for i in range(n_batches):
        logger.debug(f"Running batch {i} into model")

        batch = next(dataloader)
        forecasts = forecasts + nwp_irradiance_simple_run_one_batch(
            batch=batch, batch_idx=i, session=session
        )

    # select first 338 forecast
    if len(forecasts) > N_GSP:
        logger.debug(
            f"There are more than {N_GSP} forecasts, so just taking the first {N_GSP}. "
            "This can happen due to rounding up the examples to fit in batches"
        )
        forecasts = forecasts[0:N_GSP]

    # convert to sql objects
    forecasts_sql = [f.to_orm() for f in forecasts]

    if add_national_forecast:
        # add national forecast
        try:
            forecasts_sql.append(make_national_forecast(forecasts=forecasts, n_gsps=n_gsps))
        except Exception as e:
            logger.error(e)
            # TODO remove this
            logger.error('Did not add national forecast, carry on for the moment')

    logger.info(f"Made {len(forecasts_sql)} forecasts")

    return forecasts_sql


def nwp_irradiance_simple_run_one_batch(
    batch: Union[dict, Batch], batch_idx: int, session: Session
) -> List[Forecast]:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    # set up forecasts fields
    forecast_creation_time = datetime.now(tz=timezone.utc)

    # TODO make input data from actual data
    input_data_last_updated = make_fake_input_data_last_updated()

    # get gsp metadata
    # load metadata
    metadata = get_gsp_metadata_from_eso()
    metadata.set_index("gsp_id", drop=False, inplace=True)

    # make location x,y in osgb
    metadata["location_x"], metadata["location_y"] = lat_lon_to_osgb(
        lat=metadata["centroid_lat"], lon=metadata["centroid_lon"]
    )

    # run model
    irradiance_mean = nwp_irradiance_simple(batch)

    forecasts = []
    for i in range(batch.metadata.batch_size):

        # get gsp id from eso metadata
        meta_data_index = metadata.index[
            np.isclose(
                metadata.location_x.round(), batch.metadata.x_center_osgb[i], rtol=1e-05, atol=1e-05
            )
            & np.isclose(
                metadata.location_y.round(), batch.metadata.y_center_osgb[i], rtol=1e-05, atol=1e-05
            )
        ]

        if len(meta_data_index) == 0:
            raise Exception(
                "Could not find GSP location at"
                f"{batch.metadata.x_center_osgb[i]=}"
                f"{batch.metadata.y_center_osgb[i]=} in"
                f"{metadata.location_x=}, "
                f"{metadata.location_y=}"
            )

        gsp_ids = metadata.loc[meta_data_index].gsp_id.values

        location = get_location(gsp_id=gsp_ids[0], session=session)
        logger.debug(location)
        location = Location.from_orm(location)

        forecast_values = []
        for t_index in irradiance_mean.time_index:
            # add timezone
            target_time = (
                batch.metadata.t0_datetime_utc[i].replace(tzinfo=timezone.utc)
                + timedelta(minutes=30) * t_index.values
            )

            forecast_values.append(
                ForecastValue(
                    target_time=target_time,
                    expected_power_generation_megawatts=irradiance_mean[i, t_index],
                )
            )

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

    # take mean across all x and y dims

    irradence_mean = nwp.data.mean(axis=(2, 3))

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 100

    return irradence_mean
