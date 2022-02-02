""" Main Application """
import logging
from datetime import datetime

import click

from nowcasting_forecast import N_GSP, __version__
from nowcasting_forecast.batch import make_batches
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts, make_fake_national_forecast
from nowcasting_forecast.database.save import save
from nowcasting_forecast.models.nwp_solar_simple import nwp_irradiance_simple_run_all_batches
from nowcasting_forecast.utils import floor_30_minutes_dt

logger = logging.getLogger(__name__)


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

    logger.info(f"Running forecast app ({__version__})")

    connection = DatabaseConnection(url=db_url)
    with connection.get_session() as session:

        if fake:
            forecasts = make_dummy_forecasts()
        else:

            # make batches
            make_batches()

            # make forecasts
            forecasts = nwp_irradiance_simple_run_all_batches(session=session)

        # save forecasts
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
