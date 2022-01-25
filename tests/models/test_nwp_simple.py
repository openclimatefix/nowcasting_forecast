from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradence_simple,
    nwp_irradence_simple_run_one_batch,
)


def test_nwp_irradence_simple(batch):

    predictions = nwp_irradence_simple(batch=batch)


def test_nwp_irradence_simple_run_one_batch(batch):

    f = nwp_irradence_simple_run_one_batch(batch=batch)
