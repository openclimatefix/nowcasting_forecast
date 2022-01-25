import tempfile
from datetime import datetime
from typing import List

import numpy as np
import pytest
from sqlalchemy import event

from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.models import Base, ForecastSQL, ForecastValueSQL


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
