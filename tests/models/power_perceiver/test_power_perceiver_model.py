import os
import tempfile

import numpy as np
import pytest
import torch
from nowcasting_forecast.models.power_perceiver.model import Model

import nowcasting_forecast


def test_model_init():

    _ = Model()


def test_model_load_weights():
    model = Model()

    with tempfile.TemporaryDirectory() as tempdir:
        model_nwp_simple_trained_weights = os.path.join(tempdir, "weights.ckpt")
        torch.save({"state_dict": model.state_dict()}, model_nwp_simple_trained_weights)

        model.load_model(local_filename=model_nwp_simple_trained_weights, use_hf=False)


def test_model_load_weights_from_hf():
    model = Model.from_pretrained("openclimatefix/power_perceiver")
    model.set_gsp_id_to_one = True


def test_model_load_weights_from_hf_load_model():
    model = Model()
    model = model.load_model(use_hf=True)


def test_model_load_weights_error():
    model = Model()

    with pytest.raises(Exception):
        model.load_from_checkpoint(checkpoint_path="./error.ckpt")


def test_forward():

    # Need to make fake NumpyBatch
    model = Model(set_gsp_id_to_one=True)

    # model.forward(batch)
