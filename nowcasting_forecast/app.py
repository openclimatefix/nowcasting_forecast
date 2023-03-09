""" Main Application """
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

import click
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts, make_fake_national_forecast
from nowcasting_datamodel.save.save import save
from sqlalchemy.orm import Session

from nowcasting_forecast import N_GSP, __version__
from nowcasting_forecast.batch import make_batches
from nowcasting_forecast.models.cnn.cnn import cnn_run_one_batch
from nowcasting_forecast.models.cnn.dataloader import get_cnn_data_loader
from nowcasting_forecast.models.cnn.model import Model as CNN_Model
from nowcasting_forecast.models.nwp_simple_trained.model import Model
from nowcasting_forecast.models.nwp_simple_trained.nwp_simple_trained import (
    nwp_irradiance_simple_trained_run_one_batch,
)
from nowcasting_forecast.models.nwp_solar_simple import nwp_irradiance_simple_run_one_batch
from nowcasting_forecast.models.utils import general_forecast_run_all_batches
from nowcasting_forecast.utils import floor_minutes_dt

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)
formatter = logging.Formatter("%(levelname)-8s %(name) %(lineno)d} %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)
logging.getLogger("nowcasting_forecast").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)
logging.getLogger("nowcasting_dataset").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)
logging.getLogger("nowcasting_datamodel").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)
# TODO make logs show up in AWS


@click.command()
@click.option(
    "--db-url",
    default=None,
    envvar="DB_URL",
    help="The Database URL where forecasts will be saved",
    type=click.STRING,
)
@click.option(
    "--fake",
    default=False,
    envvar="FAKE",
    help="Option to make Fake forecasts",
    type=click.BOOL,
)
@click.option(
    "--model-name",
    default="cnn",
    envvar="MODEL_NAME",
    help="Select which model to use",
    type=click.STRING,
)
@click.option(
    "--batch-save-dir",
    default=None,
    envvar="BATCH_SAVE_DIR",
    help="Optional directory to save first bacth to",
    type=click.STRING,
)
@click.option(
    "--n-gsps",
    default=N_GSP,
    envvar="N_GSPS",
    help="Optional number of GSPs to run. Default is N_GSP",
    type=click.INT,
)
@click.option(
    "--update-national",
    default=True,
    envvar="UPDATE_NATIONAL",
    help="Update the national forecast in the latest table",
    type=click.BOOL,
)
@click.option(
    "--update-gsps",
    default=True,
    envvar="UPDATE_GSPS",
    help="Update the GSPS forecast in the latest table",
    type=click.BOOL,
)
def run(
    db_url: str,
    fake: bool = False,
    model_name: str = "nwp_simple",
    batch_save_dir: Optional[str] = None,
    n_gsps: Optional[int] = N_GSP,
    update_national: Optional[bool] = True,
    update_gsps: Optional[bool] = True,
):
    """
    Run main app.

    There is an option to make fake forecasts
    """

    logger.info(f"Running forecast app ({__version__})")

    connection = DatabaseConnection(url=db_url)
    with connection.get_session() as session:
        if fake:
            forecasts = make_dummy_forecasts(session=session, n_gsps=n_gsps)
        else:
            if model_name == "nwp_simple":
                config_filename = "nowcasting_forecast/config/mvp_v0.yaml"
            elif model_name == "nwp_simple_trained":
                config_filename = "nowcasting_forecast/config/mvp_v1.yaml"
            elif model_name == "cnn":
                config_filename = "nowcasting_forecast/config/mvp_v2.yaml"
            elif model_name == "pvnet":
                config_filename = "nowcasting_forecast/config/pvnet_v1.yaml"
            else:
                raise NotImplementedError(f"Model {model_name} has not be implemented")

            with tempfile.TemporaryDirectory() as temporary_dir:
                # make batches
                save_dir = batch_save_dir + "batch/" if batch_save_dir is not None else None
                make_batches(
                    temporary_dir=temporary_dir,
                    config_filename=config_filename,
                    batch_save_dir=save_dir,
                    n_gsps=n_gsps,
                )

                # make forecasts
                if model_name == "nwp_simple":
                    forecasts = general_forecast_run_all_batches(
                        session=session,
                        batches_dir=temporary_dir,
                        callable_function_for_on_batch=nwp_irradiance_simple_run_one_batch,
                        model_name="nwp_simple",
                        n_gsps=n_gsps,
                    )

                elif model_name == "nwp_simple_trained":
                    forecasts = general_forecast_run_all_batches(
                        session=session,
                        batches_dir=temporary_dir,
                        callable_function_for_on_batch=nwp_irradiance_simple_trained_run_one_batch,
                        model_name="nwp_simple_trained",
                        ml_model=Model,
                        n_gsps=n_gsps,
                    )
                elif model_name == "cnn":
                    dataloader = get_cnn_data_loader(
                        src_path=f"{temporary_dir}/live",
                        tmp_path=f"{temporary_dir}/live",
                        batch_save_dir=batch_save_dir,
                    )
                    forecasts = general_forecast_run_all_batches(
                        session=session,
                        batches_dir=temporary_dir,
                        callable_function_for_on_batch=cnn_run_one_batch,
                        model_name="cnn",
                        ml_model=CNN_Model,
                        dataloader=dataloader,
                        use_hf=True,
                        configuration_file="nowcasting_forecast/config/pvnet_v1.yaml",
                        n_gsps=n_gsps,
                    )
                else:
                    raise NotImplementedError(f"model name {model_name} has not be implemented. ")

        # save forecasts
        save(
            forecasts=forecasts,
            session=session,
            update_national=update_national,
            update_gsp=update_gsps,
        )


def make_dummy_forecasts(session: Session, n_gsps: Optional[int] = N_GSP):
    """Make dummy forecasts

    Dummy forecasts are made for all gsps and for several forecast horizons
    """
    # time now rounded down by 30 mins
    t0_datetime_utc = floor_minutes_dt(datetime.utcnow())

    # make gsp fake results
    forecasts = make_fake_forecasts(
        gsp_ids=list(range(1, n_gsps + 1)), t0_datetime_utc=t0_datetime_utc, session=session
    )

    # add national forecast
    forecasts.append(make_fake_national_forecast(session=session, t0_datetime_utc=t0_datetime_utc))

    return forecasts


if __name__ == "__main__":
    run()
