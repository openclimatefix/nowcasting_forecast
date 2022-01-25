from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast.models.nwp_solar_simple import nwp_irradence_simple


def test_nwp_irradence_simple():

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.process.batch_size = 4
    configuration.input_data.nwp.nwp_channels = ["dlwrf"]

    batch = Batch.fake(configuration=configuration)

    predictions = nwp_irradence_simple(batch=batch)
