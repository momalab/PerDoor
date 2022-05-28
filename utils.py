import torch
import torch.nn.functional as F

import sklearn.metrics.pairwise as smp

import logging
import numpy as np

num_clients = 100
num_selected = 10
num_rounds = 5000
epochs = 2
batch_size = 64
attacker_id = 0

source_class = 0
target_class = 2

model_path = "models/"
weight_path = "weights/"
backdoor_image_path = "backdoor_images/"

num_layers = 22

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

def fedavg(global_model, client_models, r, ni, ns):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.add(torch.stack([(ni[i]/ns)*(client_models[i].state_dict()[k].float() - global_model.state_dict()[k].float()) for i in range(len(client_models))], 0).sum(0), global_model.state_dict()[k].float())
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    torch.save(global_model, model_path+f"round_{r}_global.pth")

#Adapted from: https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/master/rules/multiKrum.py
def krum(global_model, client_models, r):
    n = num_selected
    f = 1
    topk = n - f - 2
    global_dict = global_model.state_dict()
    client_params = None
    for i in range(len(client_models)):
        params = None
        for k in global_dict.keys():
            if params is None:
                params = torch.flatten(client_models[i].state_dict()[k].float())
            else:
                params = torch.cat((params, torch.flatten(client_models[i].state_dict()[k].float())))
        if client_params is None:
            client_params = params
        else:
            client_params = torch.vstack((client_params, params))
    cdist = torch.cdist(client_params, client_params, p=2)
    nbhDist, nbh = torch.topk(cdist, topk + 1, largest=False)
    krum = torch.argmin(nbhDist.sum(1))
    for k in global_dict.keys():
        global_dict[k] = client_models[krum].state_dict()[k].float()
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    torch.save(global_model, model_path+f"round_{r}_global.pth")

#Adapted from: https://github.com/DistributedML/FoolsGold/blob/master/deep-fg/fg/foolsgold.py
def foolsgold(global_model, client_models, r):
    global_dict = global_model.state_dict()
    client_params = None
    for i in range(len(client_models)):
        params = None
        for k in global_dict.keys():
            if params is None:
                params = torch.flatten(client_models[i].state_dict()[k].float())
            else:
                params = torch.cat((params, torch.flatten(client_models[i].state_dict()[k].float())))
        if client_params is None:
            client_params = params
        else:
            client_params = torch.vstack((client_params, params))
    n_clients = num_selected
    cs = smp.cosine_similarity(client_params.cpu()) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv <= 0] = 1e-8
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv <= 0)] = 1e-8
    w = wv / wv.sum()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([w[i]*(client_models[i].state_dict()[k].float()) for i in range(len(client_models))], 0).sum(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    torch.save(global_model, model_path+f"round_{r}_global.pth")

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