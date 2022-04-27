import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts
from nowcasting_datamodel.models import ForecastSQL
from nowcasting_datamodel.models.base import Base_Forecast, Base_PV
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.data_sources.fake.batch import make_random_image_coords_osgb
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast import N_GSP
from nowcasting_forecast.utils import floor_30_minutes_dt


@pytest.fixture
def forecast_sql(db_session):

    # create
    f = make_fake_forecasts(gsp_ids=[1], session=db_session)

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def forecasts(db_session) -> List[ForecastSQL]:

    # create
    f = make_fake_forecasts(gsp_ids=list(range(0, 10)), session=db_session)

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def forecasts_all(db_session) -> List[ForecastSQL]:

    # create
    f = make_fake_forecasts(gsp_ids=list(range(0, N_GSP)), session=db_session)

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def db_connection():

    url = os.getenv("DB_URL")

    connection = DatabaseConnection(url=url)
    Base_Forecast.metadata.create_all(connection.engine)
    Base_PV.metadata.create_all(connection.engine)

    yield connection

    Base_Forecast.metadata.drop_all(connection.engine)
    Base_PV.metadata.drop_all(connection.engine)


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()


@pytest.fixture()
def configuration():

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.process.batch_size = 4
    configuration.input_data.nwp.nwp_channels = ["dlwrf"]

    return configuration


@pytest.fixture()
def batch(configuration):

    batch = Batch.fake(configuration=configuration, temporally_align_examples=True)

    return batch


@pytest.fixture()
def batch_nwp(configuration):
    configuration.input_data.nwp.history_minutes = 30
    configuration.input_data.nwp.history_minutes = 120

    batch = Batch.fake(configuration=configuration, temporally_align_examples=True)

    return batch


@pytest.fixture
def nwp_data():

    # middle of the UK
    x_center_osgb = 500_000
    y_center_osgb = 500_000
    t0_datetime_utc = t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow()) - timedelta(hours=2)
    image_size = 1000
    time_steps = 10

    x, y = make_random_image_coords_osgb(
        size=image_size, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb, km_spacing=2
    )

    # time = pd.date_range(start=t0_datetime_utc, freq="30T", periods=10)
    step = [timedelta(minutes=60 * i) for i in range(0, time_steps)]

    coords = (
        ("init_time", [t0_datetime_utc]),
        ("variable", np.array(["dswrf"])),
        ("step", step),
        ("x", x),
        ("y", y),
    )

    nwp = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(1, 1, time_steps, image_size, image_size),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!
    return nwp.to_dataset(name="UKV")
