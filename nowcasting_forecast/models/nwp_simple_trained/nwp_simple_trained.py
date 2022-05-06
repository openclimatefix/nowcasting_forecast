""" Run the nwp simple trained model

1. Setup model and load weights from s3/neptune
2. run model by looping over batches
3. format predictions
4. add national forecast
"""


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
from nowcasting_forecast.models.nwp_simple_trained.model import Model
from nowcasting_forecast.models.nwp_simple_trained.xr_utils import re_order_dims
from nowcasting_forecast.utils import floor_30_minutes_dt

logger = logging.getLogger(__name__)

NAME = "nwp_simple_trained"


def nwp_irradiance_simple_trained_run_one_batch(
    batch: Union[dict, Batch],
    session: Session,
    pytorch_model,
    n_examples: Optional[int] = None,
    input_data_last_updated: Optional[InputDataLastUpdatedSQL] = None,
) -> List[ForecastSQL]:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = Batch(**batch)

    if n_examples is None:
        n_examples = batch.metadata.batch_size

    # set up forecasts fields
    forecast_creation_time = datetime.now(tz=timezone.utc)

    # get model name
    model = get_model(name=NAME, version=nowcasting_forecast.__version__, session=session)

    if input_data_last_updated is None:
        input_data_last_updated = make_fake_input_data_last_updated()

    # run model
    predictions = nwp_irradiance_simple_trained(batch, model=pytorch_model)

    forecasts = []
    for i in range(batch.metadata.batch_size):
        if i >= n_examples:
            break

        # get id from location
        gsp_id = batch.metadata.space_time_locations[i].id

        location = get_location(gsp_id=gsp_id, session=session)

        forecast_values = []
        for t_index in range(len(predictions[0])):
            # add timezone
            target_time = (
                batch.metadata.space_time_locations[i].t0_datetime_utc.replace(tzinfo=timezone.utc)
                + timedelta(minutes=30) * t_index
            )

            forecast_values.append(
                ForecastValueSQL(
                    target_time=target_time,
                    expected_power_generation_megawatts=float(predictions[i, t_index]),
                )
            )

        forecasts.append(
            ForecastSQL(
                model=model,
                location=location,
                forecast_creation_time=forecast_creation_time,
                forecast_values=forecast_values,
                input_data_last_updated=input_data_last_updated,
            )
        )

    return forecasts


def nwp_irradiance_simple_trained(batch: Batch, model) -> xr.DataArray:
    """Predictions for one batch

    Just to take the mean NWP data across the batch

    Returns a data array of nwp irradiance with dimensions for examples and time.
    """

    nwp = batch.nwp

    nwp = re_order_dims(nwp)

    # move to torch
    nwp_batch_ml = nwp.torch.to_tensor(["data", "time", "init_time", "x_osgb", "y_osgb"])

    nwp = nwp_batch_ml["data"]

    # normalize
    dswrf_mean = 294.6696933986283
    dswrf_std = 233.1834250473355
    nwp = nwp - dswrf_mean
    nwp = nwp / dswrf_std

    # run model
    predictions = model(nwp)

    # re-normalize
    # load capacity
    capacity = pd.read_csv(
        os.path.join(os.path.dirname(nowcasting_forecast.__file__), "data", "gsp_capacity.csv"),
        index_col=["gsp_id"],
    )
    capacity = capacity.loc[batch.metadata.ids]

    # multiply predictions by capacities
    predictions = capacity.values * predictions.detach().cpu().numpy()

    return predictions
