""" CNN model with NWP, Satellite and PV trained model

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-980/monitoring


Default parameters are set from the trained model
"""

import logging
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nowcasting_dataloader.batch import BatchML
from torch import nn

from nowcasting_forecast.models.hub import NowcastingModelHubMixin

logging.basicConfig()
_LOG = logging.getLogger(__name__)


class Model(pl.LightningModule, NowcastingModelHubMixin):
    """CNN Forecast model"""

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        include_pv_or_gsp_yield_history: bool = True,
        include_nwp: bool = True,
        forecast_minutes: int = 120,  # 180 for pvent
        history_minutes: int = 30,
        number_of_conv3d_layers: int = 6,
        conv3d_channels: int = 32,
        image_size_pixels: int = 24,
        nwp_image_size_pixels: int = 64,
        number_sat_channels: int = 11,
        number_nwp_channels: int = 1,  # 4 for pvent
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
        fc3_output_features: int = 64,
        output_variable: str = "gsp_yield",
        embedding_dem: int = 8,
        include_pv_yield_history: int = True,
        include_future_satellite: int = False,
        live_satellite_images: bool = True,
        gsp_forecast_minutes: int = 480,
        gsp_history_minutes: int = 120,
        include_sun: bool = True,
    ):
        """
        3d conv model, that takes in different data streams

        architecture is roughly
        1. satellite image time series goes into many 3d convolution layers.
        2. nwp time series goes into many 3d convolution layers.
        3. Final convolutional layer goes to full connected layer. This is joined by
        other data inputs like
        - pv yield
        - time variables
        Then there ~4 fully connected layers which end up forecasting the
        pv yield / gsp into the future

        include_pv_or_gsp_yield_history: include pv yield data
        include_nwp: include nwp data
        forecast_len: the amount of minutes that should be forecasted
        history_len: the amount of historical minutes that are used
        number_of_conv3d_layers, number of convolution 3d layers that are use
        conv3d_channels, the amount of convolution 3d channels
        image_size_pixels: the input satellite image size
        nwp_image_size_pixels: the input nwp image size
        number_sat_channels: number of nwp channels
        fc1_output_features: number of fully connected outputs nodes out of the
        the first fully connected layer
        fc2_output_features: number of fully connected outputs nodes out of the
        the second fully connected layer
        fc3_output_features: number of fully connected outputs nodes out of the
        the third fully connected layer
        output_variable: the output variable to be predicted
        number_nwp_channels: The number of nwp channels there are
        include_future_satellite: option to include future satellite images, or not
        live_satellite_images: bool. Live satellite images are only available after 30 minutes,
            so lets make sure we don't use non available data in training
        """

        self.include_pv_or_gsp_yield_history = include_pv_or_gsp_yield_history
        self.include_nwp = include_nwp
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
        self.include_pv_yield_history = include_pv_yield_history
        self.include_future_satellite = include_future_satellite
        self.live_satellite_images = live_satellite_images
        self.number_sat_channels = number_sat_channels
        self.image_size_pixels = image_size_pixels
        self.gsp_forecast_minutes = gsp_forecast_minutes
        self.gsp_history_minutes = gsp_history_minutes
        self.include_sun = include_sun

        self.gsp_forecast_length = self.gsp_forecast_minutes // 30
        self.gsp_history_length = self.gsp_history_minutes // 30

        super().__init__()

        self.history_len_60 = int(np.ceil(self.history_minutes / 60))
        self.history_len_30 = int(np.ceil(self.history_minutes / 30))
        self.history_len_5 = int(np.ceil(self.history_minutes / 5))
        self.forecast_len_60 = (
            self.forecast_minutes // 60
        )  # the number of forecast timestemps for 60 minutes data
        self.forecast_len = self.forecast_minutes // 30
        self.forecast_len_5 = self.forecast_minutes // 5
        self.number_of_pv_samples_per_batch = 128
        self.number_of_samples_per_batch = 32
        self.batch_size = 32

        conv3d_channels = conv3d_channels

        if include_future_satellite:
            self.cnn_output_size_time = self.forecast_len_5 + self.history_len_5 + 1
        else:
            self.cnn_output_size_time = self.history_len_5 + 1

        if live_satellite_images:
            # remove the last 6 satellite images (30 minutes) as no available live
            self.cnn_output_size_time = self.cnn_output_size_time - 6
            if self.cnn_output_size_time <= 0:
                assert Exception("Need to use at least 30 mintues of satellite data in the past")

        self.cnn_output_size = (
            conv3d_channels
            * ((image_size_pixels - 2 * self.number_of_conv3d_layers) ** 2)
            * self.cnn_output_size_time
        )

        self.nwp_cnn_output_size = (
            conv3d_channels
            * ((nwp_image_size_pixels - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len_60 + self.history_len_60 + 1)
        )

        # conv0
        self.sat_conv0 = nn.Conv3d(
            in_channels=number_sat_channels,
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
            setattr(self, f"sat_conv{i + 1}", layer)

        self.fc1 = nn.Linear(
            in_features=self.cnn_output_size, out_features=self.fc1_output_features
        )
        self.fc2 = nn.Linear(
            in_features=self.fc1_output_features, out_features=self.fc2_output_features
        )

        # nwp
        if include_nwp:
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

        if self.embedding_dem:
            self.pv_system_id_embedding = nn.Embedding(
                num_embeddings=940, embedding_dim=self.embedding_dem
            )

        if self.include_pv_yield_history:
            self.pv_fc1 = nn.Linear(
                in_features=self.number_of_pv_samples_per_batch * (self.history_len_5 + 1),
                out_features=128,
            )
        if self.include_sun:
            # the minus 12 is bit of hard coded smudge for pvnet
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len_5 + self.history_len_5 + 1),
                out_features=16,
            )

        fc3_in_features = self.fc2_output_features
        if include_pv_or_gsp_yield_history:
            fc3_in_features += self.number_of_samples_per_batch * self.gsp_history_length
        if include_nwp:
            fc3_in_features += 128
        if self.embedding_dem:
            fc3_in_features += self.embedding_dem
        if self.include_pv_yield_history:
            fc3_in_features += 128
        if self.include_sun:
            fc3_in_features += 16

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=self.fc3_output_features)
        self.fc4 = nn.Linear(
            in_features=self.fc3_output_features, out_features=self.gsp_forecast_length
        )
        self.save_hyperparameters()

    def forward(self, batch: Union[BatchML, dict]):
        """
        Forward pass
        """

        if isinstance(batch, dict):
            batch = BatchML(**batch)

        sat_data = batch.satellite.data.float()
        nwp_data = batch.nwp.data.float()
        pv_data = batch.pv.pv_yield.float()
        gsp_data = batch.gsp.gsp_yield.float()

        # ******************* Satellite imagery *************************
        # Shape: batch_size, channel, seq_length, height, width
        # sat_data = x.satellite.data.float()
        batch_size, n_chans, seq_len, height, width = sat_data.shape

        if not self.include_future_satellite:
            sat_data = sat_data[:, :, : self.history_len_5 + 1]

        if self.live_satellite_images:
            sat_data = sat_data[:, :, :-6]

        # :) Pass data through the network :)
        out = F.relu(self.sat_conv0(sat_data))
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = getattr(self, f"sat_conv{i + 1}")
            out = F.relu(layer(out))

        out = out.reshape(batch_size, self.cnn_output_size)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # which has shape (batch_size, 128)

        # add gsp yield
        if self.include_pv_or_gsp_yield_history:
            gsp_yield_history = (
                gsp_data[:, : self.gsp_history_length, : self.number_of_samples_per_batch]
                .nan_to_num(nan=0.0)
                .float()
            )

            gsp_yield_history = gsp_yield_history.reshape(
                gsp_yield_history.shape[0], gsp_yield_history.shape[1] * gsp_yield_history.shape[2]
            )
            # join up
            out = torch.cat((out, gsp_yield_history), dim=1)

        # add the pv yield history. This can be used if trying to predict gsp
        if self.include_pv_yield_history:
            # just take the first 128
            pv_yield_history = (
                pv_data[:, : self.history_len_5 + 1, :128].nan_to_num(nan=0.0).float()
            )

            pv_yield_history = pv_yield_history.reshape(
                pv_yield_history.shape[0], pv_yield_history.shape[1] * pv_yield_history.shape[2]
            )
            pv_yield_history = F.relu(self.pv_fc1(pv_yield_history))

            out = torch.cat((out, pv_yield_history), dim=1)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # shape: batch_size, n_chans, seq_len, height, width
            # nwp_data = x.nwp.data.float()

            print(nwp_data.shape)
            out_nwp = F.relu(self.nwp_conv0(nwp_data))
            for i in range(0, self.number_of_conv3d_layers - 1):
                layer = getattr(self, f"nwp_conv{i + 1}")
                out_nwp = F.relu(layer(out_nwp))

            # fully connected layers
            out_nwp = out_nwp.reshape(batch_size, self.nwp_cnn_output_size)
            out_nwp = F.relu(self.nwp_fc1(out_nwp))
            out_nwp = F.relu(self.nwp_fc2(out_nwp))

            # join with other FC layer
            out = torch.cat((out, out_nwp), dim=1)

        # ********************** Embedding of PV system ID ********************
        if self.embedding_dem:
            id = batch.gsp.gsp_id[0 : self.batch_size, 0]

            id = id.type(torch.IntTensor)
            id = id.to(out.device)
            id_embedding = self.pv_system_id_embedding(id)
            out = torch.cat((out, id_embedding), dim=1)

        if self.include_sun:
            sun = torch.cat((batch.sun.sun_azimuth_angle, batch.sun.sun_elevation_angle), dim=1)
            out_sun = self.sun_fc1(sun)
            out = torch.cat((out, out_sun), dim=1)

        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(batch_size, self.gsp_forecast_length)

        return out

    def load_model(
        self,
        local_filename: Optional[str] = None,
        use_hf: bool = True,
    ):
        """
        Load model weights
        """

        if use_hf:
            # _LOG.debug('Loading mode from Hugging Face "openclimatefix/nowcasting_pvnet_v1" ')
            # model = Model.from_pretrained("openclimatefix/nowcasting_pvnet_v1")
            _LOG.debug('Loading mode from Hugging Face "openclimatefix/nowcasting_cnn_v5" ')
            model = Model.from_pretrained("openclimatefix/nowcasting_cnn_v5")
            _LOG.debug("Loading mode from Hugging Face: done")
            return model
        else:
            _LOG.debug(f"Loading model weights from {local_filename}")
            model = self.load_from_checkpoint(checkpoint_path=local_filename)
            _LOG.debug("Loading model weights: done")
            return model
