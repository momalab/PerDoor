import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import utils
import numpy as np
from operator import add

def extract_weights(model, r):
    if not os.path.exists(utils.weight_path):
        os.mkdir(utils.weight_path)
    model_name = f'round_{r}_global'
    model = torch.load(utils.model_path+model_name+".pth")
    params = []
    for f in model.features:
        if isinstance(f, nn.Conv2d):
            p = f.state_dict()
            weight = p['weight'].cpu().numpy()
            bias = p['bias'].cpu().numpy()
            params.append(weight)
            params.append(bias)
    for c in model.classifier:
        if isinstance(c, nn.Linear):
            p = c.state_dict()
            weight = p['weight'].cpu().numpy()
            bias = p['bias'].cpu().numpy()
            params.append(weight)
            params.append(bias)
    np.save(utils.weight_path+model_name+".npy", params)

def get_param_no_change(attacker_selected_rounds, threshold):
    params = [[] for _ in range(utils.num_layers)]
    for r in attacker_selected_rounds:
        data = np.load(utils.weight_path+f"round_{r-1}_global.npy", allow_pickle=True)
        for i in range(len(data)):
            params[i].append(data[i])
    param_no_change = []
    for i in range(len(params)):
        data = np.array(params[i])
        std = np.std(data, axis = 0)
        no_change = std < threshold
        param_no_change.append(no_change)
    return param_no_change

def get_param_not_important(model, r):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testdata = datasets.CIFAR10('./data', train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
    test_data = iter(test_loader)
    model.eval()
    grads = None
    for _ in range(10000):
        g = []
        data, target = next(test_data)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        for f in model.features:
            if isinstance(f, nn.Conv2d):
                g.append(f.weight.grad.abs())
                g.append(f.bias.grad.abs())
        for c in model.classifier:
            if isinstance(c, nn.Linear):
                g.append(c.weight.grad.abs())
                g.append(c.bias.grad.abs())
        if grads is None:
            grads = g
        else:
            grads = list(map(add, grads, g))
    grads[:] = [x / 10000 for x in grads]
    param_not_important = []
    for i in range(len(grads)):
        data = grads[i].cpu().numpy()
        threshold = np.mean(data)
        not_important = data < threshold
        param_not_important.append(not_important)
    return param_not_important

def create_non_uniform_backdoor_images(model, img, eps):
    model.eval()
    backdoor_images = []
    for item in img:
        item = torch.tensor(item).cuda()
        x_adv = Variable(item, requires_grad=True)
        for i in range(10):
            output = model(x_adv)
            loss = -F.cross_entropy(output, torch.tensor([utils.target_class]).cuda(), reduction='sum')
            loss.backward()
            x_adv = x_adv + eps*x_adv.grad.data.sign()
            x_adv = torch.clamp(x_adv, item-eps, item+eps)
            x_adv = Variable(x_adv, requires_grad=True)
        backdoor_images.append(x_adv.cpu().detach().numpy())
    np.save(utils.backdoor_image_path+f"backdoor_images.npy", backdoor_images)
    return backdoor_images

def create_backdoor_network(model, backdoor_images, r, delta, target_class):
    backdoor_params = np.load(f"backdoor_params.npy", allow_pickle=True)
    orig_params = []
    for f in model.features:
        if isinstance(f, nn.Conv2d):
            orig_params.append(f.weight)
            orig_params.append(f.bias)
    for c in model.classifier:
        if isinstance(c, nn.Linear):
            orig_params.append(c.weight)
            orig_params.append(c.bias)
    for _ in range(10):
        backdoor_modification_sign = get_backdoor_modification_sign(model, backdoor_images, target_class)
        params = []
        for f in model.features:
            if isinstance(f, nn.Conv2d):
                params.append(f.weight)
                params.append(f.bias)
        for c in model.classifier:
            if isinstance(c, nn.Linear):
                params.append(c.weight)
                params.append(c.bias)
        backdoor_done = []
        for i in range(len(params)):
            change = backdoor_modification_sign[i] * delta
            backdoor_mod = backdoor_params[i].astype(int)
            backdoored = np.clip(params[i].cpu().detach().numpy() - backdoor_mod * change, orig_params[i].cpu().detach().numpy() - delta, orig_params[i].cpu().detach().numpy() + delta)
            backdoor_done.append(backdoored)
        layers = ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25', 'classifier.0', 'classifier.2', 'classifier.4']
        model_dict = model.state_dict()
        for i in range(len(layers)):
            model_dict[layers[i]+'.weight'] = torch.tensor(backdoor_done[2*i])
            model_dict[layers[i]+'.bias'] = torch.tensor(backdoor_done[2*i+1])
        model.load_state_dict(model_dict)
    torch.save(model, utils.model_path+f"round_{r}_local_backdoor_target_{target_class}.pth")

def get_backdoor_modification_sign(model, backdoor_images, target_class):
    model.eval()
    grads = None
    for im in range(len(backdoor_images)):
        output = model(torch.tensor(backdoor_images[im]).cuda())
        g = []
        loss = F.cross_entropy(output, torch.tensor([target_class]).cuda(), reduction='sum')
        loss.backward()
        for f in model.features:
            if isinstance(f, nn.Conv2d):
                g.append(f.weight.grad)
                g.append(f.bias.grad)
        for c in model.classifier:
            if isinstance(c, nn.Linear):
                g.append(c.weight.grad)
                g.append(c.bias.grad)
        if grads is None:
            grads = g
        else:
            grads = list(map(add, grads, g))
    grads[:] = [(x / len(backdoor_images)).cpu().numpy() for x in grads]
    return grads