from nowcasting_forecast.models import Forecast, Statistic
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.save import save
from datetime import datetime, timedelta
import numpy as np

N_GSP = 338


def floor_30_minutes_dt(dt):
    """
    Floor a datetime by 30 mins.

    For example:
    2021-01-01 17:01:01 --> 2021-01-01 17:00:00
    2021-01-01 17:35:01 --> 2021-01-01 17:30:00

    :param dt:
    :return:
    """
    approx = np.floor(dt.minute / 30.0) * 30
    dt = dt.replace(minute=0)
    dt = dt.replace(second=0)
    dt = dt.replace(microsecond=0)
    dt += timedelta(minutes=approx)

    return dt

def run(url, fake: bool = False):

    if fake:
        forecasts = make_dummy_forecasts()
    else:
        raise Exception('Not implemented yet')

    connection = DatabaseConnection(url=url)
    with connection.get_session() as session:
        save(forecasts=forecasts,session=session)


def make_dummy_forecasts():
    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow())

    # make fake reuslts
    forecasts = []
    for i in range(N_GSP):
        for horizon in [0, 30, 60, 90]:
            s = Statistic(name='yhat', value=np.random.uniform(0, 1000))
            f = Forecast(t0_datetime_utc=t0_datetime_utc,
                         target_datetime_utc=t0_datetime_utc + timedelta(minutes=horizon),
                         gsp_id=i, statistics=[s])

            forecasts.append(f)

    return forecasts
