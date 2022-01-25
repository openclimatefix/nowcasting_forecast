""" Using ManagerLive to make batches """
from datetime import datetime

from nowcasting_dataset.manager.manager_live import ManagerLive

from nowcasting_forecast.utils import floor_30_minutes_dt


def make_batches(
    config_filename: str = "nowcasting_forecast/config/mvp_v0.yaml",
    t0_datetime_utc: datetime = None,
):
    """Make batches from config file"""

    if t0_datetime_utc is None:
        t0_datetime_utc = floor_30_minutes_dt(datetime.utcnow())  # add timezone

    # load config
    manager = ManagerLive()
    manager.load_yaml_configuration(config_filename, set_git=False)

    # make location file
    manager.initialize_data_sources(names_of_selected_data_sources=["gsp", "nwp"])
    manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
        t0_datetime=t0_datetime_utc
    )

    # remove gsp as a datasource
    manager.data_sources.pop("gsp")

    # make batches
    manager.create_batches()
