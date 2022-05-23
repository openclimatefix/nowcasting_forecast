""" functions to filter forecasts on the sun """
import logging
from typing import List

from nowcasting_datamodel.models import ForecastSQL
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.geospatial import calculate_azimuth_and_elevation_angle

ELEVATION_LIMIT = 0
logger = logging.getLogger(__name__)


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
