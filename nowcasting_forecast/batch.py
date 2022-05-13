""" Using ManagerLive to make batches """
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from nowcasting_dataset.manager.manager_live import ManagerLive

from nowcasting_forecast import N_GSP
from nowcasting_forecast.utils import floor_minutes_dt

logger = logging.getLogger(__name__)


def make_batches(
    config_filename: str = "nowcasting_forecast/config/mvp_v0.yaml",
    t0_datetime_utc: datetime = None,
    temporary_dir: Optional[str] = None,
    n_gsps: Optional[int] = N_GSP,
):
    """Make batches from config file"""

    logger.info(f"Making batches using configuration file: {config_filename}")

    if t0_datetime_utc is None:
        t0_datetime_utc = floor_minutes_dt(datetime.utcnow())  # add timezone

    # load config
    manager = ManagerLive()

    manager.load_yaml_configuration(config_filename, set_git=False)

    if temporary_dir is not None:
        manager.config.output_data.filepath = Path(temporary_dir)

    # over write nwp zarr path
    if (os.getenv("NWP_PATH", None) is not None) and (manager.config.input_data.nwp is not None):
        manager.config.input_data.nwp.nwp_zarr_path = os.getenv("NWP_PATH")
        logger.debug(f"WIll be opening nwp file: {manager.config.input_data.nwp.nwp_zarr_path}")

    # over write hrv sat zarr path
    if (os.getenv("HRV_SAT_PATH", None) is not None) and (
        manager.config.input_data.hrvsatellite is not None
    ):
        manager.config.input_data.hrvsatellite.hrvsatellite_zarr_path = os.getenv("HRV_SAT_PATH")
        logger.debug(
            f"WIll be opening sat file:"
            f" {manager.config.input_data.hrvsatellite.hrvsatellite_zarr_path}"
        )

    # over write sat zarr path
    if (os.getenv("SAT_PATH", None) is not None) and (
        manager.config.input_data.satellite is not None
    ):
        manager.config.input_data.satellite.satellite_zarr_path = os.getenv("SAT_PATH")
        logger.debug(
            f"WIll be opening sat file: {manager.config.input_data.satellite.satellite_zarr_path}"
        )

    # make location file
    manager.initialize_data_sources(
        names_of_selected_data_sources=["gsp", "nwp", "pv", "satellite", "hrvsatellite"]
    )
    manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
        t0_datetime=t0_datetime_utc,
        n_gsps=n_gsps,
    )

    # remove gsp as a datasource
    manager.data_sources.pop("gsp")

    # make batches
    manager.create_batches()
