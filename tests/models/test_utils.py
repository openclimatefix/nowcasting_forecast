import os
import tempfile

from nowcasting_datamodel.models import InputDataLastUpdatedSQL, LocationSQL, ForecastSQL, MLModelSQL
from nowcasting_dataset.config.save import save_yaml_configuration

from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradiance_simple,
    nwp_irradiance_simple_run_one_batch,
)

from nowcasting_forecast.models.utils import general_forecast_run_all_batches


def test_general_run_all_batches(batch, configuration, db_session, input_data_last_updated):

    with tempfile.TemporaryDirectory() as tempdir:
        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = general_forecast_run_all_batches(
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=False,
            session=db_session,
            callable_forecast_function_for_on_batch=nwp_irradiance_simple_run_one_batch,
            model_name='test_model'
        )

        assert len(f) == 4


def test_general_batches_and_national(batch, configuration, db_session, input_data_last_updated):

    n_gsps = configuration.process.batch_size

    with tempfile.TemporaryDirectory() as tempdir:

        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = general_forecast_run_all_batches(
            session=db_session,
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=True,
            n_gsps=n_gsps,
            callable_forecast_function_for_on_batch=nwp_irradiance_simple_run_one_batch,
            model_name='test_model'
        )

        assert len(f) == n_gsps + 1


def test_general_run_all_batches_check_locations(batch, configuration, db_session, input_data_last_updated):

    with tempfile.TemporaryDirectory() as tempdir:
        batch.save_netcdf(batch_i=0, path=os.path.join(tempdir, "live"))

        configuration.output_data.filepath = tempdir
        configuration_file = os.path.join(tempdir, "configuration.yaml")
        save_yaml_configuration(configuration=configuration)

        f = general_forecast_run_all_batches(
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=False,
            session=db_session,
            callable_forecast_function_for_on_batch=nwp_irradiance_simple_run_one_batch,
            model_name='test_model'
        )

        db_session.add_all(f)
        db_session.commit()

        f = general_forecast_run_all_batches(
            n_batches=1,
            configuration_file=configuration_file,
            add_national_forecast=False,
            session=db_session,
            callable_forecast_function_for_on_batch=nwp_irradiance_simple_run_one_batch,
            model_name='test_model'
        )

        db_session.add_all(f)
        db_session.commit()

        assert len(f) == 4

        assert len(db_session.query(LocationSQL).all()) == batch.metadata.batch_size
        assert len(db_session.query(InputDataLastUpdatedSQL).all()) == 1
        assert len(db_session.query(ForecastSQL).all()) == batch.metadata.batch_size * 2
        assert len(db_session.query(MLModelSQL).all()) == 1
