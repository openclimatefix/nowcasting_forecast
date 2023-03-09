import os

import torch
from nowcasting_datamodel.models import InputDataLastUpdatedSQL, LocationSQL

from nowcasting_forecast.models.nwp_simple_trained.model import Model
from nowcasting_forecast.models.nwp_simple_trained.nwp_simple_trained import (
    nwp_irradiance_simple_trained,
    nwp_irradiance_simple_trained_run_one_batch,
)
from nowcasting_forecast.models.utils import check_results_df


def save_fake_weights(path) -> str:
    model = Model()
    model_nwp_simple_trained_weights = os.path.join(path, "weights.ckpt")
    torch.save({"state_dict": model.state_dict()}, model_nwp_simple_trained_weights)

    return model_nwp_simple_trained_weights


def test_nwp_irradiance_simple(batch_nwp):
    model = Model()
    _ = nwp_irradiance_simple_trained(batch=batch_nwp, model=model)


def test_nwp_irradiance_simple_run_one_batch(batch_nwp, db_session):
    model = Model()
    f = nwp_irradiance_simple_trained_run_one_batch(batch=batch_nwp, pytorch_model=model)

    check_results_df(f)

    # make sure the target times are different
    assert f.iloc[0].target_datetime_utc != f.iloc[1].target_datetime_utc
    assert len(f) == 4 * 4  # batch size 4 and 4 values
