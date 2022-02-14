from nowcasting_forecast.models.nwp_simple_trained.model import Model
import torch


def test_model_init():

    _ = Model()


def test_model_load_weights():
    model = Model()
    model.load_model()


def test_forward():
    # shape: batch_size, n_chans, seq_len, height, width
    nwp_data = torch.rand(1, 1, 4, 64, 64)

    model = Model()
    model.forward(nwp_data=nwp_data)