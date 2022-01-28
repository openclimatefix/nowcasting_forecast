import pandas as pd

from nowcasting_forecast.database.models import Forecast, ForecastValue, Location, national_gb_label
from typing import List
from nowcasting_forecast import N_GSP


def make_national_forecast(forecasts: List[Forecast]):
    """This takes a list of forecast and adds up all teh forecast values

    Note that a different method to do this, would be to do this in the database
    """
    assert len(forecasts) == N_GSP
    # TODO check gsps are unique

    location = Location(label=national_gb_label)

    # forecast_national = forecasts[0]

    # make pandas dataframe of all the forecast values with a gsp id
    forecast_values_flat = []
    for forecast in forecasts:

        gsp_id = forecast.location.gsp_id

        one_gsp = pd.DataFrame([value.dict() for value in forecast.forecast_values])
        one_gsp["gps_id"] = gsp_id
        # this is a data frame with the follow columns
        # - gps_id
        # - target_time
        # - expected_power_generation_megawatts

        forecast_values_flat.append(one_gsp)

    forecast_values = pd.concat(forecast_values_flat)
    forecast_values[
        "count"
    ] = 1  # this will be used later to check that there are 338 forecasts for each target_time

    # group by target time
    forecast_values = forecast_values.groupby(["target_time"]).sum()
    assert (forecast_values["count"] == N_GSP).all() # could do this with 'is_unique'
    forecast_values.reset_index(inplace=True)

    # change back to ForecastValue
    forecast_values = [
        ForecastValue(**forecast_value) for forecast_value in forecast_values.to_dict("records")
    ]

    return Forecast(
        forecast_values=forecast_values,
        location=location,
        model_name=forecasts[0].model_name,
        input_data_last_updated=forecasts[0].input_data_last_updated,
        forecast_creation_time=forecasts[0].forecast_creation_time,
    )
