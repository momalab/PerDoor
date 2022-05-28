import torch
from torchvision import datasets, transforms
import utils
import numpy as np
import os

# Use this manual seed to reproduce all results
# ==================================================
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# ==================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
traindata = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / utils.num_clients) for _ in range(utils.num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=utils.batch_size, shuffle=True) for x in traindata_split]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testdata = datasets.CIFAR10('./data', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=utils.batch_size, shuffle=True)
ni = [int(traindata.data.shape[0] / utils.num_clients) for _ in range(utils.num_clients)]
ns = ni[0] * utils.num_selected