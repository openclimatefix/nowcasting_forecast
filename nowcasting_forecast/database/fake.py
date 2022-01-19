""" Functions used to make fake forecasts"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

from nowcasting_forecast.database.models import (
    ForecastSQL,
    ForecastValueSQL,
    InputDataLastUpdatedSQL,
    LocationSQL,
)


def make_fake_location(gsp_id: int) -> LocationSQL:
    """Make fake location with gsp id"""
    return LocationSQL(label=f"GSP_{gsp_id}", gsp_id=gsp_id)


def make_fake_input_data_last_updated() -> InputDataLastUpdatedSQL:
    """Make fake input data last updated"""
    now = datetime.now(tz=timezone.utc)

    return InputDataLastUpdatedSQL(gsp=now, nwp=now, pv=now, satellite=now)


def make_fake_forecast_value(target_time) -> ForecastValueSQL:
    """Make fake fprecast value"""
    return ForecastValueSQL(
        target_time=target_time, expected_power_generation_megawatts=np.random.random()
    )


def make_fake_forecast(gsp_id: int, t0_datetime_utc: Optional[datetime] = None) -> ForecastSQL:
    """Make one fake forecast"""
    location = make_fake_location(gsp_id=gsp_id)
    input_data_last_updated = make_fake_input_data_last_updated()

    if t0_datetime_utc is None:
        t0_datetime_utc = datetime(2022, 1, 1, tzinfo=timezone.utc)

    # create
    forecast_values = []
    for target_datetime_utc in [t0_datetime_utc, t0_datetime_utc + timedelta(minutes=30)]:
        f = make_fake_forecast_value(target_time=target_datetime_utc)
        forecast_values.append(f)

    forecast = ForecastSQL(
        forecast_creation_time=datetime(2022, 1, 1, tzinfo=timezone.utc),
        location=location,
        input_data_last_updated=input_data_last_updated,
        forecast_values=forecast_values,
    )

    return forecast


def make_fake_forecasts(
    gsp_ids: List[int], t0_datetime_utc: Optional[datetime] = None
) -> List[ForecastSQL]:
    """Make many fake forecast"""
    forecasts = []
    for gsp_id in gsp_ids:
        forecasts.append(make_fake_forecast(gsp_id=gsp_id, t0_datetime_utc=t0_datetime_utc))

    return forecasts