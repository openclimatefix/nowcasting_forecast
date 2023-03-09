"""
log in to the HF command line then run this command to upload a new model

'hugginface-cli login' (might need to do pip install huggingface_hub)
The whole process might take about 5 mins
"""

from nowcasting_forecast.models.cnn.model import Model

# filename = "results_w_r1042_e4.ckpt"
# filename = "w_r1171_e3.ckpt"
# filename = "w_r1171_e3.ckpt"
filename = "results_w_r1238_e8.ckpt"

model = Model.load_from_checkpoint(filename)
# model.push_to_hub("nowcasting_cnn_v2", organization="openclimatefix")
# model.push_to_hub("nowcasting_cnn_v3", organization="openclimatefix")
model.push_to_hub("nowcasting_pvnet_v1", organization="openclimatefix")
