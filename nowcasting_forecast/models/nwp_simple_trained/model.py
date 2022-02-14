""" NWP simple trained model

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-951/charts

Default parameters are set from the trained model
"""


import logging
import os
from typing import Optional

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import clip, nn

logging.basicConfig()
_LOG = logging.getLogger(__name__)


class Model(pl.LightningModule):

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        forecast_minutes: int = 120,
        history_minutes: int = 60,
        number_of_conv3d_layers: int = 6,
        conv3d_channels: int = 32,
        nwp_image_size_pixels: int = 64,
        number_nwp_channels: int = 1,
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
        fc3_output_features: int = 64,
        output_variable: str = "gsp_yield",
        embedding_dem: int = 16,
    ):
        """
        3d conv model, that takes in different data streams

        architecture is roughly
        1. nwp time series goes into many 3d convolution layers.
        2. Final convolutional layer goes to full connected layer.
        3. Then there ~4 fully connected layers which end up forecasting the pv yield / gsp into the future

        forecast_len: the amount of minutes that should be forecasted
        history_len: the amount of historical minutes that are used
        number_of_conv3d_layers, number of convolution 3d layers that are use
        conv3d_channels, the amount of convolution 3d channels
        image_size_pixels: the input satellite image size
        nwp_image_size_pixels: the input nwp image size
        number_sat_channels: number of nwp channels
        fc1_output_features: number of fully connected outputs nodes out of the the first fully connected layer
        fc2_output_features: number of fully connected outputs nodes out of the the second fully connected layer
        fc3_output_features: number of fully connected outputs nodes out of the the third fully connected layer
        output_variable: the output variable to be predicted
        number_nwp_channels: The number of nwp channels there are
        """

        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.number_of_nwp_features = 128
        self.fc1_output_features = fc1_output_features
        self.fc2_output_features = fc2_output_features
        self.fc3_output_features = fc3_output_features
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.output_variable = output_variable
        self.number_nwp_channels = number_nwp_channels
        self.embedding_dem = embedding_dem

        self.history_len_60 = int(np.ceil(self.history_minutes / 60))
        self.forecast_len_60 = (
            self.forecast_minutes // 60
        )  # the number of forecast timestemps for 60 minutes data
        self.forecast_len = self.forecast_minutes // 30

        super().__init__()

        conv3d_channels = conv3d_channels

        self.nwp_cnn_output_size = (
            conv3d_channels
            * ((nwp_image_size_pixels - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len_60 + self.history_len_60 + 1)
        )

        # nwp
        self.nwp_conv0 = nn.Conv3d(
            in_channels=number_nwp_channels,
            out_channels=conv3d_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 0, 0),
        )
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = nn.Conv3d(
                in_channels=conv3d_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            )
            setattr(self, f"nwp_conv{i + 1}", layer)

        self.nwp_fc1 = nn.Linear(
            in_features=self.nwp_cnn_output_size, out_features=self.fc1_output_features
        )
        self.nwp_fc2 = nn.Linear(
            in_features=self.fc1_output_features, out_features=self.number_of_nwp_features
        )

        fc3_in_features = self.number_of_nwp_features

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=self.fc3_output_features)
        self.fc4 = nn.Linear(in_features=self.fc3_output_features, out_features=self.forecast_len)

        # not needed
        self.history_len_5 = int(np.ceil(self.history_minutes / 5))
        self.pv_fc1 = nn.Linear(
            in_features=128 * (6 + 1),
            out_features=128,
        )
        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=940, embedding_dim=self.embedding_dem
        )

    def forward(self, nwp_data):

        # shape: batch_size, n_chans, seq_len, height, width
        nwp_data = nwp_data.data.float()
        out_nwp = F.relu(self.nwp_conv0(nwp_data))

        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = getattr(self, f"nwp_conv{i + 1}")
            out_nwp = F.relu(layer(out_nwp))

        # fully connected layers
        out_nwp = out_nwp.reshape(nwp_data.shape[0], self.nwp_cnn_output_size)
        out_nwp = F.relu(self.nwp_fc1(out_nwp))
        out = F.relu(self.nwp_fc2(out_nwp))

        # which has shape (batch_size, 128)

        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(nwp_data.shape[0], self.forecast_len)

        out = clip(out, min=0)

        return out

    def load_model(
        self, local_filename: Optional[str] = "./temp.ckpt", remote_filename: Optional[str] = None
    ):

        if remote_filename is None:
            remote_filename = "s3://nowcasting-ml-models-development/v1/predict_pv_yield_951.ckpt"

        # download weights from s3
        _LOG.debug(f"Downloading from {remote_filename} to {local_filename}")

        # remote_filename = os.path.abspath(remote_filename)
        filesystem = fsspec.open(os.path.abspath(remote_filename)).fs
        try:
            filesystem.get(remote_filename, local_filename)
        except FileNotFoundError as e:
            _LOG.error(e)
            message = f"Could not copy {remote_filename} to {local_filename}"
            _LOG.error(message)
            raise FileNotFoundError(message)

        # load weights into model
        return self.load_from_checkpoint(checkpoint_path=local_filename)
