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

### v2

Added sun features

### v3

The model can predict GSP through the night. This was done by including 
night time training data.

### v4

This model can still predict GSP, even if GSP historic data is NaN. 
