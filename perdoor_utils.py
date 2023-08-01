import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import utils
import numpy as np
from operator import add
import dataLoader
from dataLoader import get_data

def extract_weights(model, r):
    if not os.path.exists(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/"):
        os.mkdir(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/")
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
    params= np.array(params, dtype=object)
    np.save(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r}_{utils.dataset}.npy", params)

def extract_weights_resnet(model, r):
    if not os.path.exists(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/"):
        os.mkdir(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/")
    params = []
    for p in model.parameters():
        if len(p.shape) != 1:
            params.append(p.cpu().detach().numpy())
    params= np.array(params, dtype=object)
    np.save(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r}_{utils.dataset}.npy", params)

def get_param_no_change(attacker_selected_rounds, threshold):
    if utils.arch == "vgg11":
        num_layers = 22
    elif utils.arch == "vgg19":
        num_layers = 38
    elif utils.arch == "resnet18":
        num_layers = 21
    else:
        num_layers = 54
    params = [[] for _ in range(num_layers)]
    for r in attacker_selected_rounds:
        data = np.load(utils.weight_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r-1}_{utils.dataset}.npy", allow_pickle=True)
        for i in range(len(data)):
            params[i].append(data[i])
    param_no_change = []
    for i in range(len(params)):
        data = np.array(params[i])
        std = np.std(data, axis = 0)
        no_change = std < threshold
        param_no_change.append(no_change)
    return param_no_change

def get_param_not_important(model):
    _, _, testdata, _, _  = get_data()
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

def get_param_not_important_resnet(model):
    _, _, testdata, _, _  = get_data()
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
        for param in model.parameters():
            if len(param.shape) != 1:
                g.append(param.grad.abs())
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

def deepfool(image, model, overshoot, max_iter):
    f_image = model(image).flatten()
    I = torch.argsort(f_image, descending=True)
    label = I[0]
    input_shape = image.shape
    perturbed_image = torch.clone(image)
    x = Variable(perturbed_image, requires_grad=True)
    f_i = model(x)
    k_i = f_i.max(1)[1][0]
    w = torch.zeros(input_shape)
    r_tot = torch.zeros(input_shape)
    loop_i = 0
    while k_i != utils.target_class and loop_i < max_iter:
        f_i[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        f_i[0, utils.target_class].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()
        w = cur_grad - grad_orig
        f = (f_i[0, utils.target_class] - f_i[0, I[0]]).data.cpu().numpy()
        perturbation = abs(f)/np.linalg.norm(w.flatten())
        r_i = (perturbation + 1e-8) * w / np.linalg.norm(w)
        r_tot = r_tot + r_i
        perturbed_image = image + (1 + overshoot) * r_tot.cuda()
        loop_i += 1
        x = Variable(perturbed_image, requires_grad=True)
        f_i = model(x)
        k_i = f_i.max(1)[1][0]
    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, k_i, perturbed_image

def project_perturbation(data_point, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation

def create_backdoor_trigger(model, images, eps):
    _, _, testdata, _, _  = get_data()
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
    itr = 0
    max_iter_uni = 10
    num_images = 100 if len(images) > 100 else len(images)
    v = 0
    max_iter_df = 10
    overshoot = 0.02
    p = 2
    while itr < max_iter_uni: 
        for index in range(num_images):
            perturbed_inputs = (torch.tensor(images[index][None, :]) + v).cuda()
            r = model(perturbed_inputs).max(1)[1]
            if r != utils.target_class:
                dr, iter_k, k_i, pert_image = deepfool(perturbed_inputs, model, overshoot, max_iter_df)
                if iter_k < max_iter_df - 1:
                    v += dr
                    v = project_perturbation(eps, p, v)
        itr = itr + 1
    np.save(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"trigger_source_{utils.source_class}_target_{utils.target_class}_{utils.dataset}.npy", v)

def create_backdoor_network(model, images, trigger, r, delta):
    if utils.arch == "vgg11":
        layers = ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25', 'classifier.0', 'classifier.2', 'classifier.4']
    else:
        layers = ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49', 'classifier.0', 'classifier.2', 'classifier.4']
    backdoor_params = np.load(f"backdoor_params_{utils.dataset}.npy", allow_pickle=True)
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
        backdoor_modification_sign = get_backdoor_modification_sign(model, images, trigger)
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
        model_dict = model.state_dict()
        for i in range(len(layers)):
            model_dict[layers[i]+'.weight'] = torch.tensor(backdoor_done[2*i])
            model_dict[layers[i]+'.bias'] = torch.tensor(backdoor_done[2*i+1])
        model.load_state_dict(model_dict)
    torch.save(model, utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r}_local_backdoor_{utils.dataset}.pth")

def create_backdoor_network_resnet(model, images, trigger, r, delta):
    backdoor_params = np.load(f"backdoor_params_{utils.dataset}.npy", allow_pickle=True)
    orig_params = []
    for p in model.parameters():
        if len(p.shape) != 1:
            orig_params.append(p)
    for _ in range(10):
        backdoor_modification_sign = get_backdoor_modification_sign_resnet(model, images, trigger)
        params = []
        for p in model.parameters():
            if len(p.shape) != 1:
                params.append(p)
        backdoor_done = []
        for i in range(len(params)):
            change = backdoor_modification_sign[i] * delta
            backdoor_mod = backdoor_params[i].astype(int)
            backdoored = np.clip(params[i].cpu().detach().numpy() - backdoor_mod * change, orig_params[i].cpu().detach().numpy() - delta, orig_params[i].cpu().detach().numpy() + delta)
            backdoor_done.append(backdoored)
        model_dict = model.state_dict()
        model_keys = model.state_dict().keys()
        count = 0
        for item in model_keys:
            if 'conv' in item or 'shortcut.0' in item or 'linear.weight' in item:
                model_dict[item] = torch.tensor(backdoor_done[count])
                count += 1
        model.load_state_dict(model_dict)
    torch.save(model, utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r}_local_backdoor_{utils.dataset}.pth")

def get_backdoor_modification_sign(model, images, trigger):
    grads = None
    num_images = 100 if len(images) > 100 else len(images)
    for index in range(num_images):
        inputs = (torch.tensor(images[index][None, :]) + torch.tensor(trigger)).cuda()
        outputs = model(inputs)
        g = []
        loss = F.cross_entropy(outputs, torch.tensor([utils.target_class]).cuda(), reduction='sum')
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
    grads[:] = [(x / len(grads)).cpu().numpy() for x in grads]
    return grads

def get_backdoor_modification_sign_resnet(model, images, trigger):
    grads = None
    num_images = 100 if len(images) > 100 else len(images)
    for index in range(num_images):
        inputs = (torch.tensor(images[index][None, :]) + torch.tensor(trigger)).cuda()
        outputs = model(inputs)
        g = []
        loss = F.cross_entropy(outputs, torch.tensor([utils.target_class]).cuda(), reduction='sum')
        loss.backward()
        for param in model.parameters():
            if len(param.shape) != 1:
                g.append(param.grad)
        if grads is None:
            grads = g
        else:
            grads = list(map(add, grads, g))
    grads[:] = [(x / len(grads)).cpu().numpy() for x in grads]
    return grads

def find_clean_images(model):
    _, _, testdata, _, _  = get_data()
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
    clean_examples = []
    for data, act_out in test_loader:
        if act_out == utils.source_class:
            data = data.cuda()
            pred_out = model(data)
            if pred_out.argmax() == utils.source_class:
                if len(clean_examples) == 0:
                    clean_examples.append((data, pred_out))
                else:
                    index = 0
                    for i in range(len(clean_examples)):
                        if pred_out[0][utils.target_class] < clean_examples[i][1][0][utils.target_class]:
                            index = i
                            break
                    if i == len(clean_examples):
                        clean_examples = clean_examples[:index] + [(data, pred_out)]
                    else:    
                        clean_examples = clean_examples[:index] + [(data, pred_out)] + clean_examples[index:]
    images = []
    for item in clean_examples:
        images.append(item[0].cpu().numpy()[0])
    np.save(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"clean_images_source_{utils.source_class}_target_{utils.target_class}_{utils.dataset}.npy", images)