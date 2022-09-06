import os
import tempfile

import numpy as np
import pytest
import torch

import nowcasting_forecast
from power_perceiver.production.model import FullModel


def test_model_init():

    _ = FullModel()


def test_model_load_weights():
    model = FullModel()

    with tempfile.TemporaryDirectory() as tempdir:
        model_nwp_simple_trained_weights = os.path.join(tempdir, "weights.ckpt")
        torch.save({"state_dict": model.state_dict()}, model_nwp_simple_trained_weights)

        model.load_model(local_filename=model_nwp_simple_trained_weights, use_hf=False)


def test_model_load_weights_from_hf():
    model = FullModel.from_pretrained("openclimatefix/power_perceiver")
    model.set_gsp_id_to_one = True



def test_model_load_weights_from_hf_load_model():
    model = FullModel()
    model = model.load_model(use_hf=True)


def test_model_load_weights_error():
    model = FullModel()

    with pytest.raises(Exception):
        model.load_from_checkpoint(checkpoint_path="./error.ckpt")


def test_forward():

    # Need to make fake NumpyBatch
    model = FullModel(set_gsp_id_to_one=True)

    # model.forward(batch)
