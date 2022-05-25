""" Test to make one batch and run it through the model """

import os
import tempfile

import zarr

import nowcasting_forecast
from nowcasting_forecast.batch import make_batches
from nowcasting_forecast.models.cnn.cnn import cnn_run_one_batch
from nowcasting_forecast.models.cnn.dataloader import get_cnn_data_loader
from nowcasting_forecast.models.cnn.model import Model
from nowcasting_forecast.models.utils import general_forecast_run_all_batches


def test_run(
    nwp_data,
    pv_yields_and_systems,
    sat_data,
    hrv_sat_data,
    db_session,
    input_data_last_updated,
    gsp_yields_and_systems,
):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = "./temp/"
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

        make_batches(
            config_filename=os.path.join(
                os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v2.yaml"
            ),
            temporary_dir=temp_dir,
        )

        assert os.path.exists(f"{temp_dir}/live/satellite/000000.nc")
        assert os.path.exists(f"{temp_dir}/live/satellite/000001.nc")

        dataloader = get_cnn_data_loader(
            src_path=os.path.join(temp_dir, "live"), tmp_path=os.path.join(temp_dir, "live")
        )
        _ = general_forecast_run_all_batches(
            session=db_session,
            batches_dir=temp_dir,
            callable_function_for_on_batch=cnn_run_one_batch,
            model_name="nwp_simple_trained",
            ml_model=Model,
            dataloader=dataloader,
            use_hf=True,
        )
