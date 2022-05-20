""" functions to filter forecasts on the sun """
from typing import List

from nowcasting_datamodel.models import ForecastSQL
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso

ELEVATION_LIMIT = 0


def filter_forecasts_on_sun_elevation(forecasts: List[ForecastSQL]) -> List[ForecastSQL]:
    """
    Filters predictions if the sun elevation is more than X degrees below the horizon

    Args:
        forecasts: The forecast output

    Returns:
        forecast with zeroed out times
    """

    metadata = get_gsp_metadata_from_eso()
    from nowcasting_dataset.geospatial import calculate_azimuth_and_elevation_angle

    for forecast in forecasts:
        gsp_id = forecast.location.gsp_id
        lat = metadata.iloc[gsp_id].centroid_lat
        lon = metadata.iloc[gsp_id].centroid_lon
        target_times = [forecast_value.target_time for forecast_value in forecast.forecast_values]

        # get a pandas dataframe of of elevation and azimuth positions
        sun_df = calculate_azimuth_and_elevation_angle(
            latitude=lat, longitude=lon, datestamps=target_times
        )

        # check that any elevations are < 'ELEVATION_LIMIT'
        if (sun_df["elevation"] < ELEVATION_LIMIT).sum() > 0:

            # loop through target times
            for i in range(len(target_times)):
                sun_for_target_time = sun_df.iloc[i]
                if sun_for_target_time.elevation < ELEVATION_LIMIT:
                    # note sql objects are connected, so we can edit in place
                    forecast.forecast_values[i].expected_power_generation_megawatts = 0

    return forecasts
