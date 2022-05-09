import os
import tempfile

import pytest
import xarray as xr
from click.testing import CliRunner
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import Forecast, ForecastSQL, LocationSQL

from nowcasting_forecast.app import run
from nowcasting_forecast import N_GSP


def test_fake(db_connection: DatabaseConnection):

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == N_GSP + 1  # 338 gsp + national

        locations = session.query(LocationSQL).all()
        assert len(locations) == N_GSP + 1


def test_fake_twice(db_connection: DatabaseConnection):

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == (N_GSP + 1)  # 338 gsp + national

        locations = session.query(LocationSQL).all()
        assert len(locations) == N_GSP + 1

    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == (N_GSP + 1) * 2  # 338 gsp + national

        locations = session.query(LocationSQL).all()
        for l in locations:
            print(l.__dict__)
        assert len(locations) == N_GSP + 1


def test_not_fake(db_connection: DatabaseConnection, nwp_data: xr.Dataset, input_data_last_updated):

    with tempfile.TemporaryDirectory() as temp_dir:

        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        runner = CliRunner()
        response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "false"])
        assert response.exit_code == 0, response

        with db_connection.get_session() as session:
            forecasts = session.query(ForecastSQL).all()
            _ = Forecast.from_orm(forecasts[0])
            assert len(forecasts) == N_GSP + 1  # 338 gsp + national
            assert len(forecasts[0].forecast_values) > 1


@pytest.mark.skip("CI doesnt have access to AWS for model weights")
def test_mwp_1(db_connection: DatabaseConnection, nwp_data: xr.Dataset, input_data_last_updated):

    with tempfile.TemporaryDirectory() as temp_dir:

        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        runner = CliRunner()
        response = runner.invoke(
            run,
            [
                "--db-url",
                db_connection.url,
                "--fake",
                "false",
                "--model-name",
                "nwp_simple_trained",
            ],
        )
        assert response.exit_code == 0, response

        with db_connection.get_session() as session:
            forecasts = session.query(ForecastSQL).all()
            _ = Forecast.from_orm(forecasts[0])
            assert len(forecasts) == N_GSP + 1  # 338 gsp + national
            assert len(forecasts[0].forecast_values) > 1
