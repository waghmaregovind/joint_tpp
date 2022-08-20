[![PyTorch 1.11.0](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=plastic&logo=PyTorch)](https://pypi.org/project/torch/1.11.0/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes

This repository includes PyTorch implementation of the paper "Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes", CIKM 2022.

![architecture](https://github.com/waghmaregovind/joint_tpp/blob/master/figures/architecture_v2.png)


## Datasets

* Real-world datasets are available at: [`data/real`](https://github.com/waghmaregovind/joint_tpp/tree/master/data/real)
* Synthetic Hawkes datasets: Refer to [synth_data_gen.ipynb](https://github.com/waghmaregovind/joint_tpp/blob/master/code/synth_data_gen.ipynb) for Hawkes process data generation. 

## Acknowledgements

* We build on top of [`ifl-tpp` source code](https://github.com/shchur/ifl-tpp) . Thanks to [Oleksandr Shchur](https://shchur.github.io/) for the awesome codebase.

* [![`tick` library](https://img.shields.io/badge/tick-0.7.0.1-green?style=plastic)](https://github.com/X-DataInitiative/tick) and [neuralTPPs](https://github.com/babylonhealth/neuralTPPs/blob/831ed1c203c93b4e408b83b1d457af19372d6267/tpp/processes/multi_class_dataset.py#L1) are used to generate synthetic Hawkes process data. 
