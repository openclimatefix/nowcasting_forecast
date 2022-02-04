import pytest
import os
import tempfile
import xarray as xr
from click.testing import CliRunner

from nowcasting_forecast.app import run
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.models import Forecast, ForecastSQL, LocationSQL



def test_fake(db_connection: DatabaseConnection):

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == 338 + 1  # 338 gsp + national

        locations = session.query(LocationSQL).all()
        assert len(locations) == 338 + 1


@pytest.mark.skip("Skip for now #11")
def test_not_fake(db_connection: DatabaseConnection, nwp_data: xr.Dataset):

    with tempfile.TemporaryDirectory() as temp_dir:

        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_ZARR_PATH"] = nwp_path

        runner = CliRunner()
        response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "false"])
        assert response.exit_code == 0, response

        with db_connection.get_session() as session:
            forecasts = session.query(ForecastSQL).all()
            _ = Forecast.from_orm(forecasts[0])
            assert len(forecasts) == 338 + 1  # 338 gsp + national
            assert len(forecasts[0].forecast_values) > 1
