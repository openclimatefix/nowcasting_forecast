import os
import tempfile

import pytest
import torch

from nowcasting_forecast.models.cnn.model import Model


def test_model_init():

    _ = Model()

def test_model_load_weights():
    model = Model()

    with tempfile.TemporaryDirectory() as tempdir:
        model_nwp_simple_trained_weights = os.path.join(tempdir, "weights.ckpt")
        torch.save({"state_dict": model.state_dict()}, model_nwp_simple_trained_weights)

        model.load_model(remote_filename=model_nwp_simple_trained_weights)


def test_model_load_weights_error():
    model = Model()

    with pytest.raises(Exception):
        model.load_model(remote_filename="weights.ckpt")


def test_forward():
    # shape: batch_size, n_chans, seq_len, height, width
    nwp_data = torch.rand(1, 1, 4, 64, 64)
    sat_data = torch.rand(1, 11, 12, 24, 24)
    pv_data = torch.rand(1, 7, 128)
    pv_system_row_number = torch.rand(1, 2)

    model = Model()
    model.forward(nwp_data=nwp_data,sat_data=sat_data, pv_data=pv_data,pv_system_row_number=pv_system_row_number)
