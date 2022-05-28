# PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations

## Overview

## Requirements
- python 3.8.10
- Cuda compilation tools, release 10.1, V10.1.243

## Quick Setup
Create a python virtual environment and install all requirements
```
python3 -m venv perdoor
source perdoor/bin/activate
pip install -r requirements.txt
```
## Instructions
The paper describing PerDoor is available on [arXiv](https://arxiv.org/abs/2205.13523). Please refer to the paper for details of all hyperparameters.

* PerDoor implementation for unprotected FedAvg[[1]](#1) Aggregation.
```
python3 perdoor_fedavg.py --analysis_window=30 --t_delta=1e-3 --delta=1e-5 --eps=0.1
```

* PerDoor implementation for Robust Aggregation. Use `--method=krum` for _KRUM_ Aggregation and `--method=foolsgold` for _FoolsGold_ Aggregation.
```
python3 perdoor_defense.py --method=foolsgold --analysis_window=30 --t_delta=1e-3 --delta=1e-5 --eps=0.1
```

## Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam, Esha Sarkar, and Michail Maniatakos, "_PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations_", arXiv:2205.13523, 2022.

### BibTex Citation
```
@article{DBLP:journals/corr/abs-2205-13523,
    author    = {Manaar Alam and
                 Esha Sarkar and
                 Michail Maniatakos},
    title     = {{PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations}},
    journal   = {CoRR},
    volume    = {abs/2205.13523},
    year      = {2022},
    url       = {https://arxiv.org/abs/2205.13523},
    eprinttype = {arXiv},
    eprint    = {2205.13523}
}
```
## Contact Us
For more information or help with the setup, please contact Manaar Alam at alam.manaar@nyu.edu

## References
