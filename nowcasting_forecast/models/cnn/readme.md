# CNN model

This model takes both satellite and NWP video data and puts them through 
separate 3D convolutional neural networks. These are then connected with 
a few fully connected layers, joined with some simple input data like 
historic PV data. In addition, datetime features are 
added, and the position of the Sun is also used 
(‘elevation’ and ‘azimuth’ angles).

![CNN](diagram.png)

## versions

### v1 

First iteration of the model

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-1004/

### v2

Added sun features

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-1042/

### v3

The model can predict GSP through the night. This was done by including 
night time training data.

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-1171/

### v4

This model can still predict GSP, even if GSP historic data is NaN. 

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-1192/

