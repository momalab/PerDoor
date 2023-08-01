import torch
from torchvision import datasets, transforms
import utils
import numpy as np
import os

# ==================================================
# Use this manual seed to reproduce all results
# ==================================================
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# ==================================================

def get_data():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    if utils.dataset == 'mnist':
        traindata = datasets.MNIST('./data', train=True, download=True, transform=transform)
        testdata = datasets.MNIST('./data', train=False, transform=transform)
    elif utils.dataset == 'fmnist':
        traindata = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdata = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif utils.dataset == 'cifar10':
        traindata = datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar10)
        testdata = datasets.CIFAR10('./data', train=False, transform=transform_cifar10)
    else:
        traindata = datasets.CIFAR100('./data', train=True, download=True, transform=transform_cifar100)
        testdata = datasets.CIFAR100('./data', train=False, transform=transform_cifar100)
    traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / utils.num_clients) for _ in range(utils.num_clients)])
    train_loader = [torch.utils.data.DataLoader(x, batch_size=utils.batch_size, shuffle=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=utils.batch_size, shuffle=True)
    ni = [int(traindata.data.shape[0] / utils.num_clients) for _ in range(utils.num_clients)]
    ns = ni[0] * utils.num_selected
    return train_loader, test_loader, testdata, ni, ns