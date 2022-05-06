""" ML models utils functions

Pydantic models:
 - MLResult and MLResults

functions:
 - check_results_df: to check dataframe can be changed to ML Results
 - convert_to_forecast_sql: convert MLResults to ForecastSQL
"""

import logging
import numpy as np
from typing import List
import pandas as pd
import nowcasting_forecast
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic import validator
from datetime import datetime, timezone, timedelta

from nowcasting_datamodel.utils import datetime_must_have_timezone
from nowcasting_datamodel.models import (
    ForecastSQL,
    ForecastValue,
    Forecast,
    InputDataLastUpdatedSQL,
    MLModelSQL,
)
from sqlalchemy.orm.session import Session
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
    get_model,
)
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.fake import make_fake_input_data_last_updated
from nowcasting_datamodel.models import (
    Forecast,
    ForecastSQL,
    ForecastValueSQL,
    InputDataLastUpdatedSQL,
)
from nowcasting_datamodel.national import make_national_forecast
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
    get_model,
)
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.batch import Batch
from sqlalchemy.orm.session import Session

import nowcasting_forecast
from nowcasting_forecast import N_GSP
from nowcasting_forecast.dataloader import BatchDataLoader
from nowcasting_forecast.utils import floor_30_minutes_dt


logger = logging.getLogger(__name__)


class MLResult(BaseModel):
    """One results from ML models"""

    t0_datetime_utc: datetime = Field(..., description="The time when the forecast is made")
    target_datetime_utc: datetime = Field(
        ..., description="The time associated with the forecast value"
    )
    forecast_gsp_pv_outturn_mw: float = Field(..., description="The forecast value in MW")
    gsp_id: int = Field(..., description="The gsp id for the forecast")

    _normalize_t0_datetime_utc = validator("t0_datetime_utc", allow_reuse=True)(
        datetime_must_have_timezone
    )
    _normalize_target_datetime_utc = validator("target_datetime_utc", allow_reuse=True)(
        datetime_must_have_timezone
    )

    # validate forecast value >0
    @validator("forecast_gsp_pv_outturn_mw")
    def validate_forecast_gsp_pv_outturn_mw(cls, v):
        """Validate the solar_generation_kw field"""
        if v < 0:
            logger.debug(f"Changing forecast_gsp_pv_outturn_mw ({v}) to 0")
            v = 0
        return v

    # validate gsp id >=0 <= 338
    @validator("gsp_id")
    def validate_gsp_id(cls, v):
        """Validate the gsp id field"""
        if v < 0 or v > 388:
            raise Exception(f"gsp_id ({v}) is not in valid range")
        return v


class MLResults(BaseModel):
    """Group of ml results"""

    forecasts: List[MLResult] = Field(
        ...,
        description="List of forecasts for different GSPs",
    )


def check_results_df(results_df: pd.DataFrame):
    """
    Check the dataframe has the correct columns

    Args:
        results_df: results dataframe

    """

    assert len(results_df) > 0
    assert "t0_datetime_utc" in results_df.keys()
    assert "target_datetime_utc" in results_df.keys()
    assert "forecast_gsp_pv_outturn_mw" in results_df.keys()
    assert "gsp_id" in results_df.keys()


def convert_to_forecast_sql(
    results_df: pd.DataFrame,
    session: Session,
    model_name: str,
) -> List[ForecastSQL]:

    check_results_df(results_df)

    # get gsp ids
    gsps_ids = results_df["gsp_id"].unique()

    # get last input data
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    # get model name
    model = get_model(name=model_name, version=nowcasting_forecast.__version__, session=session)

    forecasts = []
    # loop over gsp ids
    for gsp_id in gsps_ids:
        results_df_one_gsp = results_df[results_df["gsp_id"] == gsp_id]

        # convert to 'MLResults'
        results = MLResults(forecast=results_df_one_gsp.to_dict(orient="records"))

        forecasts.append(convert_on_gsp_id_to_forecast_sql(
            results_for_one_gsp_id=results,
            session=session,
            input_data_last_updated=input_data_last_updated,
            ml_model=model
        ))

    return forecasts


def convert_on_gsp_id_to_forecast_sql(
    results_for_one_gsp_id: MLResults,
    session: Session,
    input_data_last_updated: InputDataLastUpdatedSQL,
    ml_model: MLModelSQL,
) -> ForecastSQL:
    """Function to convert results to forecast sql

    - gsp location is collected from database,
    - model is collected from database
    """

    # set up forecasts fields
    forecast_creation_time = datetime.now(tz=timezone.utc)

    # assert only one gsp_id
    gsps_ids = np.unique([forecast.gsp_id for forecast in results_for_one_gsp_id.forecasts])
    assert len(gsps_ids) == 1

    gsp_id = results_for_one_gsp_id.forecasts[0].gsp_id
    location = get_location(gsp_id=gsp_id, session=session)

    forecast_values = []
    for forecast in results_for_one_gsp_id.forecasts:
        # add timezone
        target_time = forecast.t0_datetime_utc.replace(tzinfo=timezone.utc)

        forecast_values.append(
            ForecastValue(
                target_time=target_time,
                expected_power_generation_megawatts=forecast.forecast_gsp_pv_outturn_mw,
            ).to_orm()
        )

    forecast = ForecastSQL(
        model=ml_model,
        location=location,
        forecast_creation_time=forecast_creation_time,
        forecast_values=forecast_values,
        input_data_last_updated=input_data_last_updated,
    )

    # validate
    _ = Forecast.from_orm(forecast)

    return forecast


def general_forecast_run_all_batches(
    session: Session,
    callable_forecast_function_for_on_batch,
    model_name: str,
    configuration_file: Optional[str] = None,
    n_batches: int = 11,
    add_national_forecast: bool = True,
    n_gsps: int = N_GSP,
    batches_dir: Optional[str] = None,
) -> List[ForecastSQL]:
    """Run model for all batches"""

    logger.info("Running nwp_irradiance_simple model")

    # time now rounded down by 30 mins
    t0_datetime_utc = floor_30_minutes_dt(datetime.now(timezone.utc))
    logger.info(f"Making forecasts for {t0_datetime_utc=}")

    # make configuration
    if configuration_file is None:
        configuration_file = os.path.join(
            os.path.dirname(nowcasting_forecast.__file__), "config", "mvp_v0.yaml"
        )
    logger.debug(f"Loading configuration {configuration_file}")
    configuration = load_yaml_configuration(filename=configuration_file)

    if batches_dir is not None:
        configuration.output_data.filepath = Path(batches_dir)

    # make dataloader
    dataloader = iter(BatchDataLoader(n_batches=n_batches, configuration=configuration))

    # loop over batch
    forecasts = []
    for i in range(n_batches):
        logger.debug(f"Running batch {i} into model")

        # calculate how many examples are needed
        n_examples = np.min([N_GSP - len(forecasts), configuration.process.batch_size])

        batch = next(dataloader)
        forecasts = forecasts + callable_forecast_function_for_on_batch(
            batch=batch,
            n_examples=n_examples,
        )

    # make into one big dataframe
    forecasts = pd.concat(forecasts)

    # select first 338 forecast
    if len(forecasts) > N_GSP:
        logger.debug(
            f"There are more than {N_GSP} forecasts, so just taking the first {N_GSP}. "
            "This can happen due to rounding up the examples to fit in batches"
        )
        forecasts = forecasts[0:N_GSP]

    # convert dataframe to ForecastSQL
    forecast_sql = convert_to_forecast_sql(results_df=forecasts, session=session,model_name=model_name)

    if add_national_forecast:
        # add national forecast
        try:
            forecast_sql.append(
                make_national_forecast(forecasts=forecast_sql, n_gsps=n_gsps, session=session)
            )
        except Exception as e:
            logger.error(e)
            # TODO remove this
            logger.error("Did not add national forecast, carry on for the moment")

    logger.info(f"Made {len(forecast_sql)} forecasts")

    return forecast_sql
