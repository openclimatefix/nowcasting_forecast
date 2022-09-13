""" Test to make one batch and run it through the model """

import os
import tempfile

import zarr

import nowcasting_forecast
from nowcasting_forecast.models.power_perceiver.dataloader import get_power_perceiver_data_loader
from nowcasting_forecast.models.power_perceiver.model import Model, power_perceiver_run_one_batch
from nowcasting_forecast.models.utils import general_forecast_run_all_batches

from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.config.save import save_yaml_configuration



def test_run(
    nwp_data,
    pv_yields_and_systems,
    sat_data,
    hrv_sat_data_2d,
    db_session,
    input_data_last_updated,
    gsp_yields_and_systems,
):

    with tempfile.TemporaryDirectory() as temp_dir:
        # save nwp data
        nwp_path = f"{temp_dir}/unittest.netcdf"
        nwp_data.to_netcdf(nwp_path, engine="h5netcdf")
        os.environ["NWP_PATH"] = nwp_path
        hrv_sat_path = f"{temp_dir}/hrv_sat_unittest.zarr.zip"
        with zarr.ZipStore(hrv_sat_path) as store:
            hrv_sat_data_2d.to_zarr(store, compute=True)
        os.environ["HRV_SAT_PATH"] = hrv_sat_path
        sat_path = f"{temp_dir}/sat_unittest.zarr.zip"
        with zarr.ZipStore(sat_path) as store:
            sat_data.to_zarr(store, compute=True)
        os.environ["SAT_PATH"] = sat_path

        topo_path = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "data", "europe_dem_2km_osgb.tif"
        )

        os.environ["TOPOGRAPHIC_FILENAME"] = topo_path

        # make configuration
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v3.yaml"
        )

        filename = temp_dir + "/configuration.yaml"
        configuration = Configuration(**load_yaml_configuration(configuration_file).__dict__)
        configuration.input_data.pv.pv_image_size_meters_height = 10000000
        configuration.input_data.pv.pv_image_size_meters_width = 10000000
        save_yaml_configuration(configuration=configuration, filename=filename)

        dataloader = get_power_perceiver_data_loader(configuration_file=filename)
        _ = general_forecast_run_all_batches(
            session=db_session,
            batches_dir=temp_dir,
            callable_function_for_on_batch=power_perceiver_run_one_batch,
            model_name="power_perceiver",
            ml_model=Model,
            dataloader=dataloader,
            use_hf=True,
            n_gsps=10,
        )
