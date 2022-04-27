import os
import tempfile

from nowcasting_forecast.batch import make_batches


def test_make_batches(nwp_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        make_batches()


def test_make_sat_batches(sat_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        sat_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["HRV_SAT_PATH"] = nwp_path

        make_batches()
