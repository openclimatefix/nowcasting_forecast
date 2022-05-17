from nowcasting_forecast.models.cnn.dataloader import get_cnn_data_loader
import os
import tempfile


from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_yaml_configuration

import nowcasting_forecast


def test_get_cnn_data_loader():

    with tempfile.TemporaryDirectory() as temp_dir:
        # make configuration
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v2.yaml"
        )
        configuration = load_yaml_configuration(filename=configuration_file)
        configuration.output_data.filepath = temp_dir

        filename = f"{temp_dir}/temp.yaml"
        save_yaml_configuration(configuration=configuration, filename=filename)

        os.mkdir(f"{temp_dir}/live")
        _ = get_cnn_data_loader(src_path="./", tmp_path="./", configuration_file=filename)
