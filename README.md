[![PyTorch 1.11.0](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=plastic&logo=PyTorch)](https://pypi.org/project/torch/1.11.0/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2210.15294-orange.svg)](https://arxiv.org/abs/2210.15294)

# Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes

This repository includes PyTorch implementation of the paper "Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes", CIKM 2022.

![architecture](https://github.com/waghmaregovind/joint_tpp/blob/master/figures/architecture_v2.png)


## Datasets

* Real-world datasets are available at: [`data/real`](https://github.com/waghmaregovind/joint_tpp/tree/master/data/real)
* Synthetic Hawkes datasets: Refer to [synth_data_gen.ipynb](https://github.com/waghmaregovind/joint_tpp/blob/master/code/synth_data_gen.ipynb) for Hawkes process data generation. 

## Acknowledgements

* We build on top of [`ifl-tpp` source code](https://github.com/shchur/ifl-tpp). [`dpp`](https://github.com/waghmaregovind/joint_tpp/tree/master/code/dpp) package is copied and modified to support joint modeling of time and mark. Thanks to [Oleksandr Shchur](https://shchur.github.io/) for the awesome codebase.

* [![`tick` library](https://img.shields.io/badge/tick-0.7.0.1-green?style=plastic)](https://github.com/X-DataInitiative/tick) and [neuralTPPs](https://github.com/babylonhealth/neuralTPPs/blob/831ed1c203c93b4e408b83b1d457af19372d6267/tpp/processes/multi_class_dataset.py#L1) are used to generate synthetic Hawkes process data. 

## Citation
```
@inproceedings{10.1145/3511808.3557399,
author = {Waghmare, Govind and Debnath, Ankur and Asthana, Siddhartha and Malhotra, Aakarsh},
title = {Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557399},
doi = {10.1145/3511808.3557399},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {1986–1995},
numpages = {10},
keywords = {multivariate temporal point processes, probabilistic modeling},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
