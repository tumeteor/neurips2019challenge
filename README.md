# neurips2019challenge
Solution for our Neurips 2019 challenge on [traffic movie forecasting](https://www.iarai.ac.at/traffic4cast/conference-programme-2019/) - top-6 teams.

# Background 
## Dataset
The high-resolution traffic map videos for the challenge were derived from positions reported by a
large fleet of probe vehicles over 12 months, and are based on over 100 billion probe points. Each
video frame summarizes GPS trajectories (of 3 channels, encode speed, volume, and direction of
traffic) mapped to spatio-temporal cells, sampled every 5 minutes and of size 495 ˆ 436. 

## Requirements
* [Tensorflow](https://www.tensorflow.org/) version >= 1.14.1
* Python version >= 3.6
* For training new models, you'll need an NVIDIA GPU

# Getting started
* `trainer.py` for training the model
## License
Apache 2.0
