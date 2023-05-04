import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import xarray as xr
import zarr
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_yaml_configuration
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite

from nowcasting_forecast.batch import make_batches
from nowcasting_forecast.utils import floor_minutes_dt


def test_make_batches(nwp_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        make_batches(temporary_dir=temp_dir, n_gsps=10)


def test_make_batches_mvp_v1(nwp_data, pv_yields_and_systems):
    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path

        now_5 = (
            pd.Timestamp(datetime.now(tz=timezone.utc)).floor("5T").replace(tzinfo=None).isoformat()
        )

        make_batches(
            config_filename="nowcasting_forecast/config/mvp_v1.yaml",
            temporary_dir=temp_dir,
            n_gsps=10,
        )

        assert os.path.exists(f"{temp_dir}/live")
        assert os.path.exists(f"{temp_dir}/live/spatial_and_temporal_locations_of_each_example.csv")

        locations = pd.read_csv(
            f"{temp_dir}/live/spatial_and_temporal_locations_of_each_example.csv"
        )
        assert pd.to_datetime(locations.iloc[0].t0_datetime_utc).isoformat() == now_5


def test_make_batches_mvp_v2_just_sat_data(sat_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        configuration = load_yaml_configuration("nowcasting_forecast/config/mvp_v2.yaml")

        configuration.input_data.nwp = None
        configuration.input_data.pv = None
        configuration.input_data.gsp.is_live = False
        configuration.input_data.gsp.metadata_only = True
        configuration.input_data.sun = None
        configuration.input_data.hrvsatellite = None

        filename = f"{temp_dir}/temp.yaml"
        save_yaml_configuration(configuration=configuration, filename=filename)

        sat_path = f"{temp_dir}/sat_unittest.zarr.zip"
        with zarr.ZipStore(sat_path) as store:
            sat_data.to_zarr(store, compute=True)
        os.environ["SAT_PATH"] = sat_path

        make_batches(config_filename=filename, temporary_dir=temp_dir, n_gsps=10)

        # open pv files and check there is something in there
        assert os.path.exists(f"{temp_dir}/live")
        assert os.path.exists(f"{temp_dir}/live/satellite")
        assert os.path.exists(f"{temp_dir}/live/satellite/000000.nc")
        sat = xr.load_dataset(f"{temp_dir}/live/satellite/000000.nc", engine="h5netcdf")
        _ = Satellite(sat)


def test_make_batches_pvnet_v1(
    nwp_data, pv_yields_and_systems, sat_data, hrv_sat_data, gsp_yields_and_systems
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

        now_5 = (
            pd.Timestamp(datetime.now(tz=timezone.utc))
            .floor("30T")
            .replace(tzinfo=None)
            .isoformat()
        )
        now_30 = (
            pd.Timestamp(datetime.now(tz=timezone.utc))
            .floor("30T")
            .replace(tzinfo=None)
            .isoformat()
        )

        make_batches(
            config_filename="nowcasting_forecast/config/pvnet_v1.yaml",
            temporary_dir=temp_dir,
            n_gsps=10,
        )

        # open pv files and check there is something in there
        assert os.path.exists(f"{temp_dir}/live")
        assert os.path.exists(f"{temp_dir}/live/nwp")
        assert os.path.exists(f"{temp_dir}/live/pv")
        assert os.path.exists(f"{temp_dir}/live/gsp")
        assert os.path.exists(f"{temp_dir}/live/satellite")
        assert os.path.exists(f"{temp_dir}/live/hrvsatellite")
        assert os.path.exists(f"{temp_dir}/live/gsp/000000.nc")
        pv = xr.load_dataset(f"{temp_dir}/live/pv/000000.nc", engine="h5netcdf")
        pv = PV(pv)
        # assert pv.power_mw.max() > 0
        assert pd.to_datetime(pv.time.values[0, -1]).isoformat() == now_5

        gsp = xr.load_dataset(f"{temp_dir}/live/gsp/000000.nc", engine="h5netcdf")
        gsp = GSP(gsp)
        assert len(gsp.time.values[0]) == 5
        assert pd.to_datetime(gsp.time.values[0, -1]).isoformat() == now_30
