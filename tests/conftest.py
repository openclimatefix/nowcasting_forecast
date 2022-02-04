import os
import tempfile
from typing import List

import xarray as xr
import numpy as np
from datetime import datetime, timedelta

import pytest
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast import N_GSP
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.models import Base, ForecastSQL
from nowcasting_dataset.data_sources.fake.batch import make_random_image_coords_osgb


@pytest.fixture
def forecast_sql(db_session):

    # create
    f = make_fake_forecasts(gsp_ids=[1])

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def forecasts(db_session) -> List[ForecastSQL]:

    # create
    f = make_fake_forecasts(gsp_ids=list(range(0, 10)))

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def forecasts_all(db_session) -> List[ForecastSQL]:

    # create
    f = make_fake_forecasts(gsp_ids=list(range(0, N_GSP)))

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def db_connection():

    # url = os.getenv("DB_URL")
    with tempfile.TemporaryDirectory() as temp_dir:

        url = f"sqlite:///{temp_dir}/test.db"

        connection = DatabaseConnection(url=url)
        Base.metadata.create_all(connection.engine)

        yield connection

        Base.metadata.drop_all(connection.engine)


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


@pytest.fixture
def nwp_data():

    # middle of the UK
    x_center_osgb = 500_000
    y_center_osgb = 500_000
    t0_datetime_utc = datetime(2022, 2, 4, 8)
    image_size = 1000
    time_steps = 10

    x, y = make_random_image_coords_osgb(
        size=image_size, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb, km_spaceing=2
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
