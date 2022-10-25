import os
import tempfile

import numpy as np
import xarray as xr
import zarr
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.config.save import save_yaml_configuration
from ocf_datapipes.utils.consts import BatchKey

import nowcasting_forecast
from nowcasting_forecast.models.power_perceiver.dataloader import get_power_perceiver_data_loader


def test_get_power_perceiver_data_loader(
    nwp_data,
    pv_yields_and_systems,
    hrv_sat_data_2d,
    db_session,
    input_data_last_updated,
    gsp_yields_and_systems,
):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.zarr.zip"
        with zarr.ZipStore(nwp_path) as store:
            nwp_data.to_zarr(store, compute=True)
        os.environ["NWP_ZARR_PATH"] = nwp_path

        hrv_sat_path = f"{temp_dir}/hrv_sat_unittest.zarr.zip"
        with zarr.ZipStore(hrv_sat_path) as store:
            hrv_sat_data_2d.to_zarr(store, compute=True)
        os.environ["HRV_SATELLITE_ZARR_PATH"] = hrv_sat_path

        topo_path = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "data", "europe_dem_2km_osgb.tif"
        )
        os.environ["TOPOGRAPHIC_FILENAME"] = topo_path

        # make configuration
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "pp_v1.yaml"
        )

        # for testing lets make sure there are plently of PV system in the area of intrested
        filename = temp_dir + "/configuration.yaml"
        configuration = Configuration(**load_yaml_configuration(configuration_file).__dict__)
        configuration.input_data.pv.pv_image_size_meters_height = 10000000
        configuration.input_data.pv.pv_image_size_meters_width = 10000000
        configuration.input_data.pv.n_pv_systems_per_example = 8
        save_yaml_configuration(configuration=configuration, filename=filename)

        data_loader = get_power_perceiver_data_loader(configuration_file=filename)

        batch = next(data_loader)

        assert BatchKey.hrvsatellite_actual in batch.keys()
        assert (
            len(batch[BatchKey.hrvsatellite_actual][:, :12, 0].shape) == 4
        )  # (example, time, y, x)

        assert batch[BatchKey.pv].shape == (4, 37, 8)  # (example, time, y, x)

        assert batch[BatchKey.hrvsatellite_time_utc].shape == (4, 37)
        assert batch[BatchKey.hrvsatellite_time_utc].shape == (
            4,
            37,
        )  # 12 history + now + 24 future = 19
        assert batch[BatchKey.nwp_target_time_utc].shape == (4, 10)
        assert batch[BatchKey.nwp_init_time_utc].shape == (4, 10)
        assert batch[BatchKey.pv_time_utc].shape == (4, 37)
        assert batch[BatchKey.gsp_time_utc].shape == (4, 19)  # 12 history + now + 6 future

        assert batch[BatchKey.hrvsatellite_actual].shape == (
            4,
            13,
            1,
            128,
            256,
        )  # 2nd dim is 12 history + now
        assert batch[BatchKey.nwp].shape == (
            4,
            10,
            9,
            4,
            4,
        )  # 2nd dim is 1 history + now + 9 future ?? TODO check
        assert batch[BatchKey.gsp].shape == (4, 19, 1)  # 2nd dim is 4 history + now + 2 future
        assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)
