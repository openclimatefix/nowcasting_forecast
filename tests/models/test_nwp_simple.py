import os
import tempfile

from nowcasting_forecast import N_GSP
from nowcasting_dataset.config.save import save_yaml_configuration

from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradiance_simple,
    nwp_irradiance_simple_run_all_batches,
    nwp_irradiance_simple_run_one_batch,
)


def test_nwp_irradiance_simple(batch):

    _ = nwp_irradiance_simple(batch=batch)


def test_nwp_irradiance_simple_run_one_batch(batch):

    f = nwp_irradiance_simple_run_one_batch(batch=batch, batch_idx=0)

    # make sure the target times are different
    assert f[0].forecast_values[0].target_time != f[0].forecast_values[1].target_time


def test_nwp_irradiance_simple_run_all_batches(batch, configuration):

    with tempfile.TemporaryDirectory() as tempdir:
        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = nwp_irradiance_simple_run_all_batches(
            n_batches=1, configuration_file=configuration_file, add_national_forecast=False
        )

        assert len(f) == 4


def test_nwp_irradiance_simple_run_all_batches_and_national(batch, configuration):

    n_gsps = configuration.process.batch_size

    with tempfile.TemporaryDirectory() as tempdir:

        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = nwp_irradiance_simple_run_all_batches(
            n_batches=1, configuration_file=configuration_file, add_national_forecast=True, n_gsps=n_gsps
        )

        assert len(f) == n_gsps + 1
