import os
import tempfile

from nowcasting_dataset.config.save import save_yaml_configuration

from nowcasting_forecast import N_GSP
from nowcasting_forecast.database.models import InputDataLastUpdatedSQL, LocationSQL
from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradiance_simple,
    nwp_irradiance_simple_run_all_batches,
    nwp_irradiance_simple_run_one_batch,
)


def test_nwp_irradiance_simple(batch):

    _ = nwp_irradiance_simple(batch=batch)


def test_nwp_irradiance_simple_run_one_batch(batch, db_session):

    f = nwp_irradiance_simple_run_one_batch(batch=batch, session=db_session)

    # make sure the target times are different
    assert f[0].forecast_values[0].target_time != f[0].forecast_values[1].target_time


def test_nwp_irradiance_simple_check_locations(batch, db_session):

    f = nwp_irradiance_simple_run_one_batch(batch=batch, session=db_session)
    db_session.add_all(f)
    db_session.commit()

    f = nwp_irradiance_simple_run_one_batch(batch=batch, session=db_session)
    db_session.add_all(f)
    db_session.commit()

    assert len(db_session.query(LocationSQL).all()) == batch.metadata.batch_size
    assert len(db_session.query(InputDataLastUpdatedSQL).all()) == 2


def test_nwp_irradiance_simple_run_all_batches(batch, configuration, db_session):

    with tempfile.TemporaryDirectory() as tempdir:
        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = nwp_irradiance_simple_run_all_batches(
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=False,
            session=db_session,
        )

        assert len(f) == 4


def test_nwp_irradiance_simple_run_all_batches_and_national(batch, configuration, db_session):

    n_gsps = configuration.process.batch_size

    with tempfile.TemporaryDirectory() as tempdir:

        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = nwp_irradiance_simple_run_all_batches(
            session=db_session,
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=True,
            n_gsps=n_gsps,
        )

        assert len(f) == n_gsps + 1
