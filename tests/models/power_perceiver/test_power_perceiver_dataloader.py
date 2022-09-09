import os
import tempfile
import zarr

import numpy as np
import xarray as xr

import nowcasting_forecast
from nowcasting_forecast.models.power_perceiver.dataloader import get_power_perceiver_data_loader

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.save import save_yaml_configuration


def test_get_power_perceiver_data_loader(
    nwp_data,
    pv_yields_and_systems,
    hrv_sat_data,
    db_session,
    input_data_last_updated,
    gsp_yields_and_systems,
):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.zarr.zip"
        with zarr.ZipStore(nwp_path) as store:
            nwp_data.to_zarr(store,  compute=True)
        os.environ["NWP_ZARR_PATH"] = nwp_path

        hrv_sat_path = f"{temp_dir}/hrv_sat_unittest.zarr.zip"
        with zarr.ZipStore(hrv_sat_path) as store:
            hrv_sat_data.to_zarr(store, compute=True)
        os.environ["HRV_SATELLITE_ZARR_PATH"] = hrv_sat_path

        topo_path = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "data", "europe_dem_2km_osgb.tif"
        )
        os.environ["TOPOGRAPHIC_FILENAME"] = topo_path

        # make configuration
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v3.yaml"
        )

        # for testing lets make sure there are plently of PV system in the area of intrested
        filename = temp_dir + "/configuration.yaml"
        configuration = Configuration(**load_yaml_configuration(configuration_file).__dict__)
        configuration.input_data.pv.pv_image_size_meters_height = 10000000
        configuration.input_data.pv.pv_image_size_meters_width = 10000000
        save_yaml_configuration(configuration=configuration, filename=filename)

        datalaoder = get_power_perceiver_data_loader(configuration_file=filename)

        _ = next(datalaoder)
