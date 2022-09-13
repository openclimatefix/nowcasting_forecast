import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts, make_fake_input_data_last_updated
from nowcasting_datamodel.models import (
    ForecastSQL,
    GSPYield,
    Location,
    LocationSQL,
    PVSystem,
    PVSystemSQL,
    PVYield,
    solar_sheffield_passiv,
)
from nowcasting_datamodel.models.base import Base_Forecast, Base_PV
from nowcasting_datamodel.models.models import StatusSQL
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.data_sources.fake.batch import make_image_coords_osgb
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast import N_GSP
from nowcasting_forecast.utils import floor_minutes_dt


@pytest.fixture
def status(db_session):
    status = StatusSQL(message="", status="ok")
    db_session.add(status)


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
    f = make_fake_forecasts(gsp_ids=list(range(1, 11)), session=db_session)

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def forecasts_all(db_session) -> List[ForecastSQL]:

    # create
    f = make_fake_forecasts(gsp_ids=list(range(1, N_GSP + 1)), session=db_session)

    # add
    db_session.add_all(f)

    return f


@pytest.fixture
def db_connection():

    url = os.getenv("DB_URL", "sqlite:///test.db")
    os.environ["DB_URL_PV"] = url
    os.environ["DB_URL"] = url

    connection = DatabaseConnection(url=url, echo=False)
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
    t0_datetime_utc = pd.Timestamp(
        floor_minutes_dt(datetime.utcnow(), minutes=60) - timedelta(hours=2)
    )
    image_size = 1000
    time_steps = 11
    init_times = 11

    x, y = make_image_coords_osgb(
        size_x=image_size,
        size_y=image_size,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        km_spacing=2,
    )

    # time = pd.date_range(start=t0_datetime_utc, freq="30T", periods=10)
    step = [timedelta(minutes=60 * i) for i in range(0, time_steps)]
    t0_datetime_utcs = [t0_datetime_utc + timedelta(minutes=60 * i) for i in range(0, init_times)]

    coords = (
        ("init_time", t0_datetime_utcs),
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
                size=(len(t0_datetime_utcs), 1, time_steps, image_size, image_size),
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
        provider=solar_sheffield_passiv,
        status_interval_minutes=5,
        longitude=-1.293,
        latitude=51.76,
        installed_capacity_kw=123,
        ml_capacity_kw=123,
    ).to_orm()
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=0,
        latitude=56,
        installed_capacity_kw=124,
        ml_capacity_kw=124,
    ).to_orm()

    t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=2)

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
            pv_yield_1.pv_system = pv_system_sql_1
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


@pytest.fixture()
def gsp_yields_and_systems(db_session):
    """Create gsp yields and systems

    gsp systems: One systems
    GSP yields:
        For system 1, gsp yields from 2 hours ago to 8 in the future at 30 minutes intervals
        For system 2: 1 gsp yield at 16.00
    """

    # this pv systems has same coordiantes as the first gsp
    gsp_yield_sqls = []
    locations = []
    for i in range(N_GSP):
        location_sql_1: LocationSQL = Location(
            gsp_id=i + 1,
            label=f"GSP_{i+1}",
            installed_capacity_mw=123.0,
        ).to_orm()

        t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=2)

        gsp_yield_sqls = []
        for hour in range(0, 10):
            for minute in range(0, 60, 30):
                datetime_utc = t0_datetime_utc + timedelta(hours=hour - 2, minutes=minute)
                gsp_yield_1 = GSPYield(
                    datetime_utc=datetime_utc,
                    solar_generation_kw=20 + hour + minute,
                ).to_orm()
                gsp_yield_1.location = location_sql_1
                gsp_yield_sqls.append(gsp_yield_1)
                locations.append(location_sql_1)

    # add to database
    db_session.add_all(gsp_yield_sqls + locations)

    db_session.commit()

    return {
        "gsp_yields": gsp_yield_sqls,
        "gs_systems": locations,
    }


@pytest.fixture()
def sat_data():

    # middle of the UK
    t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=2)

    times = [t0_datetime_utc]
    # this means there will be about 30 mins of no data.
    # This reflects the true satellite consumer
    time_steps = 20
    for i in range(1, time_steps):
        times.append(t0_datetime_utc + timedelta(minutes=5 * i))

    local_path = os.path.dirname(os.path.abspath(__file__))
    x, y = np.load(f"{local_path}/sat_data/geo.npy", allow_pickle=True)

    coords = (
        ("time", times),
        ("x_geostationary", x),
        ("y_geostationary", y),
        (
            "variable",
            np.array(
                [
                    "IR_016",
                    "IR_039",
                    "IR_087",
                    "IR_097",
                    "IR_108",
                    "IR_120",
                    "IR_134",
                    "VIS006",
                    "VIS008",
                    "WV_062",
                    "WV_073",
                ]
            ),
        ),
    )

    sat = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(time_steps, 615, 298, 11),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!

    area_attr = np.load(f"{local_path}/sat_data/area.npy")
    sat.attrs["area"] = area_attr

    sat["x_osgb"] = sat.x_geostationary
    sat["y_osgb"] = sat.y_geostationary

    return sat.to_dataset(name="data").sortby("time")


@pytest.fixture()
def hrv_sat_data_general():

    # middle of the UK
    t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=2)
    times = [t0_datetime_utc]
    time_steps = 26
    for i in range(1, time_steps):
        times.append(t0_datetime_utc + timedelta(minutes=5 * i))
    local_path = os.path.dirname(os.path.abspath(__file__))

    x, y = np.load(f"{local_path}/sat_data/hrv_geo.npy", allow_pickle=True)

    coords = (
        ("time", times),
        ("x_geostationary", x),
        ("y_geostationary", y),
        ("variable", np.array(["HRV"])),
    )

    sat = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(time_steps, 1843, 891, 1),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!
    area_attr = np.load(f"{local_path}/sat_data/hrv_area.npy")
    sat.attrs["area"] = area_attr

    return sat


@pytest.fixture()
def hrv_sat_data(hrv_sat_data_general):

    sat = hrv_sat_data_general

    sat["x_osgb"] = sat.x_geostationary
    sat["y_osgb"] = sat.y_geostationary
    return sat.to_dataset(name="data").sortby("time")


@pytest.fixture()
def hrv_sat_data_2d(hrv_sat_data_general):
    """
    Add 2d x_osgb to the satellite images
    """

    sat = hrv_sat_data_general

    xx, yy = np.meshgrid(sat.x_geostationary.values, sat.y_geostationary.values, indexing="ij")

    sat.__setitem__("x_osgb", xr.DataArray(xx, dims=["x_geostationary", "y_geostationary"]))
    sat.__setitem__(
        "y_osgb", xr.DataArray(yy[::-1, :], dims=["x_geostationary", "y_geostationary"])
    )

    # cheat to make coorindates work
    sat.x_osgb[0, 0] += 1
    sat.y_osgb[0, 0] -= 1

    assert sat.x_osgb[0, 0] < sat.x_osgb[0, -1]
    assert sat.y_osgb[0, 0] > sat.y_osgb[-1, 0]

    return sat.to_dataset(name="data").sortby("time")


@pytest.fixture()
def input_data_last_updated(db_session):
    i = make_fake_input_data_last_updated()
    db_session.add(i)
    db_session.commit()
