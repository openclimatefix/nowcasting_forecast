import os
import tempfile

import pytest
import torch
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataset.config.load import load_yaml_configuration

import nowcasting_forecast
from nowcasting_forecast.models.cnn.model import Model


def test_model_init():

    _ = Model()


def test_model_load_weights():
    model = Model()

    with tempfile.TemporaryDirectory() as tempdir:
        model_nwp_simple_trained_weights = os.path.join(tempdir, "weights.ckpt")
        torch.save({"state_dict": model.state_dict()}, model_nwp_simple_trained_weights)

        model.load_model(local_filename=model_nwp_simple_trained_weights, use_hf=False)


def test_model_load_weights_from_hf():
    model = Model.from_pretrained("openclimatefix/nowcasting_cnn_v2")


def test_model_load_weights_from_hf_load_model():
    model = Model()
    model = model.load_model(use_hf=True)


def test_model_load_weights_error():
    model = Model()

    with pytest.raises(Exception):
        model.load_from_checkpoint(checkpoint_path="./error.ckpt")


def test_forward():

    # make configuration
    configuration_file = os.path.join(
        os.path.dirname(nowcasting_forecast.__file__), "config", "cnn_v1.yaml"
    )

    configuration = load_yaml_configuration(filename=configuration_file)

    batch = BatchML.fake(configuration=configuration)

    model = Model()
    model.forward(batch)
