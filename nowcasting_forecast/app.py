""" Main Application """
from datetime import datetime

import click

from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts, make_fake_national_forecast
from nowcasting_forecast.database.save import save
from nowcasting_forecast.utils import floor_30_minutes_dt

N_GSP = 338


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

    # make gsp fake results
    forecasts = make_fake_forecasts(gsp_ids=list(range(N_GSP)), t0_datetime_utc=t0_datetime_utc)

    # add national forecast
    forecasts.append(make_fake_national_forecast(t0_datetime_utc=t0_datetime_utc))

    return forecasts


if __name__ == "__main__":
    run()
