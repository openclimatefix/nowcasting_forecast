from nowcasting_datamodel.models import InputDataLastUpdatedSQL, LocationSQL

from nowcasting_forecast.models.nwp_solar_simple import (
    nwp_irradiance_simple,
    nwp_irradiance_simple_run_one_batch,
)


def test_nwp_irradiance_simple(batch):
    _ = nwp_irradiance_simple(batch=batch)


def test_nwp_irradiance_simple_run_one_batch(batch, db_session):
    forecast_df = nwp_irradiance_simple_run_one_batch(batch=batch)

    # make sure the target times are different
    assert forecast_df.iloc[0].target_datetime_utc != forecast_df.iloc[1].target_datetime_utc
