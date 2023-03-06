""" functions to filter forecasts on the sun """
import logging
from datetime import datetime, timezone
from typing import List

from nowcasting_datamodel.models import ForecastSQL, StatusSQL
from nowcasting_datamodel.read.read import get_latest_status
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.geospatial import calculate_azimuth_and_elevation_angle
from sqlalchemy.orm.session import Session

ELEVATION_LIMIT = 0
DROP_ELEVATION_LIMIT = 5
logger = logging.getLogger(__name__)

WARNING_MESSAGE = (
    "New forecasts are offline until sunrise.  "
    "We are working on a fix that will deliver forecasts in the pre-sunrise hours."
)


def filter_forecasts_on_sun_elevation(forecasts: List[ForecastSQL]) -> List[ForecastSQL]:
    """
    Filters predictions if the sun elevation is more than X degrees below the horizon

    Args:
        forecasts: The forecast output

    Returns:
        forecast with zeroed out times
    """

    logger.info("Filtering forecasts on sun elevation")

    metadata = get_gsp_metadata_from_eso()

    for forecast in forecasts:
        gsp_id = forecast.location.gsp_id
        gsp_metadata = metadata[metadata.gsp_id == gsp_id].iloc[0]
        lat = gsp_metadata.centroid_lat
        lon = gsp_metadata.centroid_lon
        target_times = [forecast_value.target_time for forecast_value in forecast.forecast_values]

        # get a pandas dataframe of of elevation and azimuth positions
        sun_df = calculate_azimuth_and_elevation_angle(
            latitude=lat, longitude=lon, datestamps=target_times
        )

        # check that any elevations are < 'ELEVATION_LIMIT'
        if (sun_df["elevation"] < ELEVATION_LIMIT).sum() > 0:
            logger.debug(f"Got sun angle for {lat} {lon} {target_times}, and some are below zero")

            # loop through target times
            for i in range(len(target_times)):
                sun_for_target_time = sun_df.iloc[i]
                if sun_for_target_time.elevation < ELEVATION_LIMIT:
                    # note sql objects are connected, so we can edit in place
                    forecast.forecast_values[i].expected_power_generation_megawatts = 0

        else:
            logger.debug("No elevations below zero, so need to filter")

    logger.info("Done sun filtering")

    return forecasts


def drop_forecast_on_sun_elevation(forecasts: List[ForecastSQL], session: Session):
    """
    Drop forecast if UK centroid elevation is below a limit

    Idea is that right now the models have not be trained at night.
    Lets not save any forecast to the database while the
    model does know how to do night time.
    In addition lets set the status to warning with a message

    """

    # lat and lon of Oxford
    lat = 51.7520
    lon = -1.2577

    now = datetime.now(tz=timezone.utc)

    sun = calculate_azimuth_and_elevation_angle(latitude=lat, longitude=lon, datestamps=[now])
    sun_elevation = sun["elevation"].iloc[0]

    if sun_elevation < DROP_ELEVATION_LIMIT:
        logger.debug(f"Dropping all forecasts as sun elevation is {sun_elevation}")
        forecasts = []

        # get status
        status = get_latest_status(session=session)

        if (status.status != "warning") and (status.message != WARNING_MESSAGE):
            # set status
            logger.debug("Setting status to warning")
            status = StatusSQL(message=WARNING_MESSAGE, status="warning")
            session.add(status)
            session.commit()

        else:
            logger.debug(f"Warning already set to {WARNING_MESSAGE}")

    else:
        # get status
        status = get_latest_status(session=session)
        if status is not None:
            logger.debug(f"Status is {status.__dict__}")

            # set status to ok
            if (status.status == "warning") and (status.message == WARNING_MESSAGE):
                logger.debug("Setting status to ok")
                status = StatusSQL(message="", status="ok")
                session.add(status)
                session.commit()

    return forecasts
