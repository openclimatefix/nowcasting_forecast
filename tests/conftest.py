import os
import tempfile
from typing import List

import pytest
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast import N_GSP
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.models import Base, ForecastSQL

# TODO #4 add test postgres database, might need to do this with docker-composer


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


# @pytest.fixture
# def db_connection():
#
#     url = os.getenv('DB_URL')
#
#     with tempfile.NamedTemporaryFile(suffix="db") as tmp:
#         url = f"sqlite:///{tmp.name}.db"
#
#         connection = DatabaseConnection(url=url)
#         Base.metadata.create_all(connection.engine)
#
#         yield connection

@pytest.fixture
def db_connection():

    url = os.getenv('DB_URL')

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
