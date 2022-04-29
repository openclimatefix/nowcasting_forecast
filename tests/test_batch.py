import os
import tempfile

import xarray as xr
from nowcasting_dataset.data_sources.pv.pv_model import PV

from nowcasting_forecast.batch import make_batches


def test_make_batches(nwp_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        make_batches(temporary_dir=temp_dir)


def test_make_batches_mvp_v1(nwp_data, pv_yields_and_systems):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        make_batches(
            config_filename="nowcasting_forecast/config/mvp_v1.yaml", temporary_dir=temp_dir
        )


def test_make_batches_mvp_v2(nwp_data, pv_yields_and_systems, sat_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path
        sat_path = f"{temp_dir}/sat_unittest.netcdf"
        sat_data.to_netcdf(sat_path, engine="h5netcdf")
        os.environ["HRV_SAT_PATH"] = sat_path

        make_batches(
            config_filename="nowcasting_forecast/config/mvp_v2.yaml", temporary_dir=temp_dir
        )

        # open pv files and check there is something in there
        assert os.path.exists(f"{temp_dir}/live")
        assert os.path.exists(f"{temp_dir}/live/nwp")
        assert os.path.exists(f"{temp_dir}/live/pv")
        pv = xr.load_dataset(f"{temp_dir}/live/pv/000000.nc", engine="h5netcdf")
        pv = PV(pv)
        assert pv.power_mw.max() > 0


