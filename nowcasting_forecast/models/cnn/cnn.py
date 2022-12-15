""" Run the CNN model

1. Setup model and load weights from XXX
2. run model by looping over batches
3. format predictions
4. add national forecast
"""


import logging
import os
from datetime import timedelta, timezone
from typing import Optional, Union

import pandas as pd
from nowcasting_dataloader.batch import BatchML

import nowcasting_forecast

logger = logging.getLogger(__name__)

NAME = "cnn"


def cnn_run_one_batch(
    batch: Union[dict, BatchML],
    pytorch_model,
    n_examples: Optional[int] = None,
) -> pd.DataFrame:
    """Run model for one batch"""

    # make sure its a Batch object
    if type(batch) == dict:
        batch = BatchML(**batch)

    if n_examples is None:
        n_examples = batch.metadata.batch_size

    # run model
    predictions = pytorch_model(batch)

    # re-normalize
    # load capacity
    capacity = pd.read_csv(
        os.path.join(os.path.dirname(nowcasting_forecast.__file__), "data", "gsp_capacity.csv"),
        index_col=["gsp_id"],
    )
    capacity = capacity.loc[batch.metadata.id]

    # multiply predictions by capacities
    predictions = capacity.values * predictions.detach().cpu().numpy()

    logger.debug(f"The maximum predictions is {predictions.max()}")
    logger.debug(f"The minimum predictions is {predictions.min()}")

    forecasts = []
    for i in range(batch.metadata.batch_size):
        if i >= n_examples:
            break

        # get id from location
        gsp_id = batch.metadata.id[i]

        # t0 value value, make sure its rounded down to the nearest 30 minutes
        t0_datetime_utc = pd.Timestamp(
            batch.metadata.t0_datetime_utc[i].replace(tzinfo=timezone.utc)
        ).ceil("30T")

        # its 12.32, t0 will be 12.30 and the predictions will be for 13.00
        if t0_datetime_utc.minute in [0, 30]:
            t0_datetime_utc = t0_datetime_utc + timedelta(minutes=30)

        if i == 0:
            logger.debug(f'The first target_time will be {t0_datetime_utc}')

        for t_index in range(len(predictions[0])):
            # add timezone
            target_time = t0_datetime_utc + timedelta(minutes=30) * t_index

            forecasts.append(
                dict(
                    t0_datetime_utc=t0_datetime_utc,
                    target_datetime_utc=target_time,
                    forecast_gsp_pv_outturn_mw=float(predictions[i, t_index]),
                    gsp_id=gsp_id,
                )
            )

    forecasts = pd.DataFrame(forecasts)

    return forecasts
