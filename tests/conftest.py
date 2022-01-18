import tempfile
from datetime import datetime
from sqlalchemy import event

import numpy as np

import pytest

from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.models import Base, ForecastSQL, StatisticSQL
from nowcasting_forecast.models import Forecast, Statistic


@pytest.fixture
def forecast_sql(db_session):

    # create
    f = ForecastSQL(
        t0_datetime_utc=datetime(2022, 1, 1), target_datetime_utc=datetime(2022, 1, 1), gsp_id=1
    )

    s0 = StatisticSQL(name="yhat", value=50.0, forecast=f)
    s1 = StatisticSQL(name="p10", value=10.0, forecast=f)

    # add
    db_session.add(f)

    return f


@pytest.fixture
def db_connection():

    with tempfile.NamedTemporaryFile(suffix="db") as tmp:
        url = f"sqlite:///{tmp.name}.db"
        connection = DatabaseConnection(url=url)
        Base.metadata.create_all(connection.engine)

        yield connection


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    with db_connection.get_session() as s:
        s.begin()
        yield s

        s.rollback()