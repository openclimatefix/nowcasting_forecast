""" Functions used to make fake forecasts"""
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pytest

from nowcasting_forecast.database.models import Base, ForecastSQL, StatisticSQL
from nowcasting_forecast.database.save import save
from nowcasting_forecast.models import Forecast, Statistic


def make_fake_forecasts(session, created_utc: Optional[datetime] = None) -> List[Forecast]:
    """
    Make foake forecasts and save them to database

    :param created_utc: optional to provide the 'created_utc' datetime
    :param session: database session
    """
    r = list(np.random.random(100))

    # create
    forecasts = []
    for gsp_id in range(0, 10):
        for t0_datetime_utc in [datetime(2022, 1, 1), datetime(2022, 1, 2)]:
            for target_datetime_utc in [t0_datetime_utc, t0_datetime_utc + timedelta(minutes=30)]:
                f = Forecast(
                    t0_datetime_utc=t0_datetime_utc,
                    target_datetime_utc=target_datetime_utc,
                    gsp_id=gsp_id,
                    statistics=[
                        Statistic(name="yhat", value=r.pop()),
                        Statistic(name="p10", value=r.pop()),
                    ],
                )
                if created_utc is not None:
                    f.created_utc = created_utc

                forecasts.append(f)

    save(forecasts=forecasts, session=session)

    return forecasts
