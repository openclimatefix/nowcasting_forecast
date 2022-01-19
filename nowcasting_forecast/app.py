""" Main Application """
from datetime import datetime, timedelta

import click
import numpy as np

from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.save import save

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


@click.command()
@click.option(
    "--db-url",
    default=None,
    envvar="DB_URL",
    help="The Database URL where forecasts will be saved",
    type=click.STRING,
)
@click.option(
    "--fake",
    default=False,
    envvar="FAKE",
    help="Option to make Fake forecasts",
    type=click.BOOL,
)
def run(db_url: str, fake: bool = False):
    """
    Run main app.

    There is an option to make fake forecasts
    """

    if fake:
        forecasts = make_dummy_forecasts()
    else:
        # 1. load data
        # 2. Make data into examples
        raise Exception("Not implemented yet")

    connection = DatabaseConnection(url=db_url)
    with connection.get_session() as session:
        save(forecasts=forecasts, session=session)


def make_dummy_forecasts():
    """Make dummy forecasts

    Dummy forecasts are made for all gsps and for several forecast horizons
    """
    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow())

    # make fake reuslts
    forecasts = make_fake_forecasts(gsp_ids=range(N_GSP), t0_datetime_utc=t0_datetime_utc)

    return forecasts


if __name__ == "__main__":
    run()
