import os
import tempfile

import xarray as xr
from click.testing import CliRunner
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import Forecast, ForecastSQL, LocationSQL, national_gb_label

from nowcasting_forecast.app import run

# def test_fake(db_connection: DatabaseConnection):
#
#     runner = CliRunner()
#     response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
#     assert response.exit_code == 0
#
#     with db_connection.get_session() as session:
#         forecasts = session.query(ForecastSQL).all()
#         _ = Forecast.from_orm(forecasts[0])
#         assert len(forecasts) == 338 + 1  # 338 gsp + national
#
#         locations = session.query(LocationSQL).order_by(LocationSQL.gsp_id).all()
#         assert len(locations) == 338 + 1
#         assert locations[0].label == national_gb_label


def test_not_fake(
    db_connection: DatabaseConnection, nwp_data: xr.Dataset, input_data_last_updated_sql
):

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
            # assert len(forecasts) == 338 + 1  # 338 gsp + national
            assert len(forecasts[0].forecast_values) > 1

            locations = session.query(LocationSQL).order_by(LocationSQL.gsp_id).all()
            assert len(locations) == 338 + 1
            assert locations[0].label == national_gb_label
            assert locations[0].gsp_id == 0
