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

    forecasts = []
    for i in range(batch.metadata.batch_size):
        if i >= n_examples:
            break

        # get id from location
        gsp_id = batch.metadata.id[i]

        # t0 value value
        t0_datetime_utc = batch.metadata.t0_datetime_utc[i].replace(tzinfo=timezone.utc)

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
