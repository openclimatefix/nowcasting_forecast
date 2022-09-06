import os
import tempfile

import nowcasting_forecast
from nowcasting_forecast.models.power_perceiver.dataloader import get_power_perceiver_data_loader


def test_get_power_perceiver_data_loader():

    with tempfile.TemporaryDirectory() as temp_dir:
        # make configuration
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v3.yaml"
        )

        _ = get_power_perceiver_data_loader(configuration_file=configuration_file)
