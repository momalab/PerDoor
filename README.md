# PerDoor: Persistent Backdoors in Federated Learning using Adversarial Perturbations

## Overview
PerDoor is a persistent-by-construction backdoor injection technique in a Federated Learning (FL) framework. PerDoor uses adversarial perturbation and targets parameters of the centralized FL model that deviate less in successive FL rounds and contribute the least to the main task accuracy for injecting backdoors. This repository presents codes to validate PerDoor in an image classification scenario using four datasets (MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100) and four CNN architectures (VGG11, VGG19, ResNet18, and ResNet34).

## Requirements
- python 3.10.9
- Cuda compilation tools, release 11.5, V11.5.119

## Quick Setup
Create a python virtual environment and install all requirements
```
python3 -m venv perdoor
source perdoor/bin/activate
pip install -r requirements.txt
```
## Instructions
The paper describing PerDoor is available [here](https://ieeexplore.ieee.org/abstract/document/10189281). Please refer to the paper for details of all hyperparameters.

* PerDoor implementation for FedAvg Aggregation [[1]](#1).
```
python3 main.py --analysis_window=<value> --t_delta=<value> --eps=<value> --delta=<value> --dataset=<dataset> --num_classes=<value> --source_class=<value> --target_class=<value> --arch=<architecture>
```

## Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam, Esha Sarkar, and Michail Maniatakos, "_PerDoor: Persistent Backdoors in Federated Learning using Adversarial Perturbations_", IEEE COINS, 2023.

### BibTex Citation
```
@inproceedings{DBLP:conf/coins/alam,
  author       = {Manaar Alam and
                  Esha Sarkar and
                  Michail Maniatakos},
  title        = {{PerDoor: Persistent Backdoors in Federated Learning using Adversarial Perturbations}},
  booktitle    = {{IEEE} International Conference on Omni-layer Intelligent Systems,
                  {COINS} 2023, Berlin, Germany, July 23-25, 2023},
  pages        = {1--6},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/COINS57856.2023.10189281},
  doi          = {10.1109/COINS57856.2023.10189281},
}
```
## Contact Us
For more information or help with the setup, please contact Manaar Alam at: alam.manaar@nyu.edu

## References
<a id="1">[1]</a> Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas, "_Communication-Efficient Learning of Deep Networks from Decentralized Data_", AISTATS 2017.