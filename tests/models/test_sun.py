from datetime import datetime

from nowcasting_forecast.models.sun import filter_forecasts_on_sun_elevation


def test_filter_forecasts_on_sun_elevation(forecasts):

    # night time
    forecasts[0].forecast_values[0].target_time = datetime(2022, 1, 1)
    forecasts[0].forecast_values[0].expected_power_generation_megawatts = 1

    # day time
    forecasts[0].forecast_values[1].target_time = datetime(2022, 1, 1, 12)
    forecasts[0].forecast_values[1].expected_power_generation_megawatts = 1

    _ = filter_forecasts_on_sun_elevation(forecasts)

    assert forecasts[0].forecast_values[0].expected_power_generation_megawatts == 0
    assert forecasts[0].forecast_values[1].expected_power_generation_megawatts == 1
