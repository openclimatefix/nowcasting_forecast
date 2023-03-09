import os
import tempfile

import pytest
import xarray as xr
import zarr
from click.testing import CliRunner
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import Forecast, ForecastSQL, LocationSQL
from nowcasting_datamodel.fake import make_fake_me_latest

from nowcasting_forecast import N_GSP
from nowcasting_forecast.app import run


def test_fake(db_connection, me_latest):
    runner = CliRunner()
    response = runner.invoke(
        run,
        ["--db-url", db_connection.url, "--fake", "true", "--n-gsps", "10"],
        catch_exceptions=True,
    )
    if response.exception:
        raise response.exception
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == (10 + 1) * 2  # 10 gsp + national, x2 for historic ones too

        locations = session.query(LocationSQL).all()
        assert len(locations) == 10 + 1


def test_fake_twice(db_connection: DatabaseConnection, me_latest):
    runner = CliRunner()
    response = runner.invoke(
        run,
        ["--db-url", db_connection.url, "--fake", "true", "--n-gsps", "10"],
        catch_exceptions=True,
    )
    if response.exception:
        raise response.exception

    assert response.exit_code == 0, response.exception

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])

        assert len(forecasts) == (10 + 1) * 2  # 338 gsp + national, x2 for historic ones too

        locations = session.query(LocationSQL).all()
        assert len(locations) == 10 + 1

    response = runner.invoke(
        run, ["--db-url", db_connection.url, "--fake", "true", "--n-gsps", "10"]
    )
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == (10 + 1) * 3  # 338 gsp + national

        locations = session.query(LocationSQL).all()
        assert len(locations) == 10 + 1


def test_not_fake(
    db_connection: DatabaseConnection,
    nwp_data: xr.Dataset,
    input_data_last_updated,
    sat_data,
    hrv_sat_data,
    gsp_yields_and_systems,
    pv_yields_and_systems,
    me_latest,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path
        hrv_sat_path = f"{temp_dir}/hrv_sat_unittest.zarr.zip"
        with zarr.ZipStore(hrv_sat_path) as store:
            hrv_sat_data.to_zarr(store, compute=True)
        os.environ["HRV_SAT_PATH"] = hrv_sat_path
        sat_path = f"{temp_dir}/sat_unittest.zarr.zip"
        with zarr.ZipStore(sat_path) as store:
            sat_data.to_zarr(store, compute=True)
        os.environ["SAT_PATH"] = sat_path

        runner = CliRunner()
        response = runner.invoke(
            run, ["--db-url", db_connection.url, "--fake", "false", "--n-gsps", "10"]
        )
        assert response.exit_code == 0, response

        with db_connection.get_session() as session:
            forecasts = session.query(ForecastSQL).all()
            _ = Forecast.from_orm(forecasts[0])
            assert len(forecasts) == (10 + 1) * 2  # 317 gsp + national, x2 for historic ones too
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
                "--n-gsps",
                "10",
            ],
        )
        assert response.exit_code == 0, response

        with db_connection.get_session() as session:
            forecasts = session.query(ForecastSQL).all()
            _ = Forecast.from_orm(forecasts[0])
            assert len(forecasts) == 10 + 1  # 10 gsp + national
            assert len(forecasts[0].forecast_values) > 1
