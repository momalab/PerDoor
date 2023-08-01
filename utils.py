import torch
import torch.nn.functional as F

import sklearn.metrics.pairwise as smp

import logging
import numpy as np

num_clients = 100
num_selected = 10
num_rounds = 1000
epochs = 2
batch_size = 64
attacker_id = 42

source_class = None
target_class = None
dataset = None
num_classes = None
arch = None
model_path = "models"
weight_path = "weights"
image_path = "images"

def client_update(client_model, optimizer, train_loader, client_index):
    client_model.train()
    for e in range(epochs):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    logging.info(f'Client {client_index} loss: {loss.item()}')
    print(f'Client {client_index} loss: {loss.item()}')
    return loss.item()

def fedavg(global_model, client_models, ni, ns):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.add(torch.stack([(ni[i]/ns)*(client_models[i].state_dict()[k].float() - global_model.state_dict()[k].float()) for i in range(len(client_models))], 0).sum(0), global_model.state_dict()[k].float())
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc