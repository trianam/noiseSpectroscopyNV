# noiseSpectroscopyNV
Source code and data for the experiments reported in the paper: **"Deep learning enhanced noise spectroscopy of a spin qubit environment"** ([http://arxiv.org/abs/2301.05079](http://arxiv.org/abs/2301.05079)).

The data folders are `data` for the synthetic datasets and `experimentalData` for the experimental data. `configurations.py` contains all the configurations for the training of ML models who are launched with `runPytorch.py` for single training and with `tuneRunPytorch.py` for the configurations that use Tune for the hyperparameters optimization.
