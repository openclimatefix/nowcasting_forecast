import os
import tempfile

import torch
from nowcasting_datamodel.models import InputDataLastUpdatedSQL, LocationSQL
from nowcasting_dataset.config.save import save_yaml_configuration

from nowcasting_forecast.models.nwp_simple_trained.model import Model
from nowcasting_forecast.models.nwp_simple_trained.nwp_simple_trained import (
    nwp_irradiance_simple_trained,
    nwp_irradiance_simple_trained_run_one_batch,
)


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
    f = nwp_irradiance_simple_trained_run_one_batch(
        batch=batch_nwp, session=db_session, pytorch_model=model
    )

    # make sure the target times are different
    assert f[0].forecast_values[0].target_time != f[0].forecast_values[1].target_time


def test_nwp_irradiance_simple_check_locations(batch_nwp, db_session):
    model = Model()
    f = nwp_irradiance_simple_trained_run_one_batch(
        batch=batch_nwp, session=db_session, pytorch_model=model
    )
    db_session.add_all(f)
    db_session.commit()

    f = nwp_irradiance_simple_trained_run_one_batch(
        batch=batch_nwp, session=db_session, pytorch_model=model
    )
    db_session.add_all(f)
    db_session.commit()

    assert len(db_session.query(LocationSQL).all()) == batch_nwp.metadata.batch_size
    assert len(db_session.query(InputDataLastUpdatedSQL).all()) == 2

