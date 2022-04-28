import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts
from nowcasting_datamodel.models import ForecastSQL, PVSystem, PVSystemSQL, PVYield
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
    os.environ["DB_URL_PV"] = url

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
    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow()) - timedelta(hours=2)
    image_size = 1000
    time_steps = 10

    x, y = make_random_image_coords_osgb(
        size_x=image_size,
        size_y=image_size,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        km_spacing=2,
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


@pytest.fixture()
def pv_yields_and_systems(db_session):
    """Create pv yields and systems

    Pv systems: Two systems
    PV yields:
        FOr system 1, pv yields from 2 hours ago to 4 in the future at 5 minutes intervals
        For system 2: 1 pv yield at 16.00
    """

    # this pv systems has same coordiantes as the first gsp
    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=-1.293,
        latitude=51.76,
    ).to_orm()
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2, provider="pvoutput.org", status_interval_minutes=5, longitude=0, latitude=56
    ).to_orm()

    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow()) - timedelta(hours=2)

    pv_yield_sqls = []
    for hour in range(0, 6):
        for minute in range(0, 60, 5):
            datetime_utc = t0_datetime_utc + timedelta(hours=hour - 2, minutes=minute)
            pv_yield_1 = PVYield(
                datetime_utc=datetime_utc,
                solar_generation_kw=hour + minute / 100,
            ).to_orm()
            if datetime_utc.hour in [22, 23, 0, 1, 2]:
                pv_yield_1.solar_generation_kw = 0
            pv_yield_1.pv_system = pv_system_sql_1s
            pv_yield_sqls.append(pv_yield_1)

    pv_yield_4 = PVYield(datetime_utc=datetime(2022, 1, 1, 4), solar_generation_kw=4).to_orm()
    pv_yield_4.pv_system = pv_system_sql_2
    pv_yield_sqls.append(pv_yield_4)

    # add to database
    db_session.add_all(pv_yield_sqls)
    db_session.add(pv_system_sql_1)
    db_session.add(pv_system_sql_2)

    db_session.commit()

    return {
        "pv_yields": pv_yield_sqls,
        "pv_systems": [pv_system_sql_1, pv_system_sql_2],
    }


@pytest.fixture
def sat_data():

    # middle of the UK
    x_center_osgb = 500_000
    y_center_osgb = 500_000
    t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow()) - timedelta(hours=2)
    times = [t0_datetime_utc]
    for i in range(1,6):
        times.append(t0_datetime_utc + timedelta(minutes=5 * i))
    image_size = 128
    time_steps = 10

    x, y = make_random_image_coords_osgb(
        size_x=image_size,
        size_y=image_size,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        km_spacing=2,
    )

    coords = (
        ("time", times),
        ("variable", np.array(["HRV"])),
        ("x_geostationary", x),
        ("y_geostationary", y),
    )

    sat = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(6, time_steps, image_size, image_size),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!
    return sat.to_dataset(name="data")
