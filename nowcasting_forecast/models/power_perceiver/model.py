""" Run the Power Perceiver model

1. Setup model and load weights from XXX
2. run model by looping over batches
3. format predictions
4. add national forecast
"""


import logging
import os
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from ocf_datapipes.utils.consts import BatchKey
from power_perceiver.production.model import FullModel
from power_perceiver.pytorch_modules.mixture_density_network import get_distribution

import nowcasting_forecast

logger = logging.getLogger(__name__)

NAME = "power_perceiver"


def power_perceiver_run_one_batch(
    batch: dict[BatchKey, np.ndarray],
    pytorch_model: FullModel,
) -> pd.DataFrame:
    """Run model for one batch"""

    # TODO Remove when the model is retrained
    pytorch_model.set_gsp_id_to_one = True

    n_examples = batch[BatchKey.hrvsatellite_actual].shape[0]
    history_idx = batch[BatchKey.gsp_t0_idx][0]
    # run model
    network_output = pytorch_model(batch)
    distribution = get_distribution(network_output["predicted_gsp_power"][history_idx + 1 :])
    predictions = distribution.mean

    # re-normalize
    # load capacity
    capacity = pd.read_csv(
        os.path.join(os.path.dirname(nowcasting_forecast.__file__), "data", "gsp_capacity.csv"),
        index_col=["gsp_id"],
    )
    capacity = capacity.loc[batch[BatchKey.gsp_id]]

    # multiply predictions by capacities
    predictions = capacity.values * predictions.detach().cpu().numpy()

    logger.debug(f"The maximum predictions is {predictions.max()}")
    logger.debug(f"The minimum predictions is {predictions.min()}")

    forecasts = []
    for i in range(n_examples):

        # get id from location
        # TODO These are all currently set to 1 as part of the change to new GSPs
        gsp_id = batch[BatchKey.gsp_id][i]

        # t0 value value
        t0_datetime_utc = pd.to_datetime(
            batch[BatchKey.gsp_time_utc][batch[BatchKey.gsp_t0_idx][i]], utc=True
        ).replace(tzinfo=timezone.utc)

        for t_index in range(len(predictions[0])):
            # add timezone
            target_time = t0_datetime_utc + timedelta(minutes=30) * (t_index + 1)

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