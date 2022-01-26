import os
import tempfile

from nowcasting_dataset.config.save import save_yaml_configuration

from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradiance_simple,
    nwp_irradiance_simple_run_all_batches,
    nwp_irradiance_simple_run_one_batch,
)


def test_nwp_irradence_simple(batch):

    _ = nwp_irradiance_simple(batch=batch)


def test_nwp_irradiance_simple_run_one_batch(batch):

    f = nwp_irradiance_simple_run_one_batch(batch=batch, batch_idx=0)


def test_nwp_irradiance_simple_run_all_batches(batch, configuration):

    with tempfile.TemporaryDirectory() as tempdir:
        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = nwp_irradiance_simple_run_all_batches(
            n_batches=1, configuration_file=configuration_file
        )
