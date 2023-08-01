import torch
import torch.optim as optim

import numpy as np
import os
import logging
from tqdm import tqdm
import argparse

import utils
import perdoor_utils
from vgg_model import VGG
import resnet_model
from dataLoader import get_data

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--analysis_window', action='store', type=int, required=True)
parser.add_argument('--t_delta', action='store', type=float, required=True)
parser.add_argument('--eps', action='store', type=float, required=True)
parser.add_argument('--delta', action='store', type=float, required=True)
parser.add_argument('--dataset', action='store', type=str, required=True)
parser.add_argument('--num_classes', action='store', type=int, required=True)
parser.add_argument('--source_class', action='store', type=int, required=True)
parser.add_argument('--target_class', action='store', type=int, required=True)
parser.add_argument('--arch', action='store', type=str, required=True)
args = parser.parse_args()

utils.dataset = args.dataset
utils.source_class = args.source_class
utils.target_class = args.target_class
utils.arch = args.arch
logging.basicConfig(filename=f'perdoor_fedavg_{utils.dataset}_{utils.arch}.log', level=logging.INFO, format='%(message)s')

if not os.path.exists(utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"):
    os.mkdir(utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/")
if not os.path.exists(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/"):
    os.mkdir(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/")

if utils.dataset == "mnist" or utils.dataset == "fmnist":
    channels = 1
else:
    channels = 3
if utils.arch == "vgg11":
    global_model = VGG('VGG11', args.num_classes, channels).cuda()
    client_models = [VGG('VGG11', args.num_classes, channels).cuda() for _ in range(utils.num_selected)]
elif utils.arch == "vgg19":
    global_model = VGG('VGG19', args.num_classes, channels).cuda()
    client_models = [VGG('VGG19', args.num_classes, channels).cuda() for _ in range(utils.num_selected)]
elif utils.arch == "resnet18":
    global_model = resnet_model.ResNet18(args.num_classes, channels).cuda()
    client_models = [resnet_model.ResNet18(args.num_classes, channels).cuda() for _ in range(utils.num_selected)]
else:
    global_model = resnet_model.ResNet34(args.num_classes, channels).cuda()
    client_models = [resnet_model.ResNet34(args.num_classes, channels).cuda() for _ in range(utils.num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

train_loader, test_loader_batch, testdata, ni, ns = get_data()

attacker_selected_rounds = []
backdoor_inserted = False
for r in range(1, utils.num_rounds+1):
    print('%d-th round' % r)
    logging.info('----------------------')
    logging.info(f'Round: {r}')
    logging.info('----------------------')
    client_idx = np.random.permutation(utils.num_clients)[:utils.num_selected]
    logging.info('Selected Clients for Update: '+str(client_idx))
    print('Selected Clients for Update: '+str(client_idx))
    if not backdoor_inserted:
        adversary_selected = utils.attacker_id in client_idx
        if not adversary_selected:
            loss = 0
            for i in tqdm(range(utils.num_selected)):
                loss += utils.client_update(client_models[i], opt[i], train_loader[client_idx[i]], client_idx[i])
            utils.fedavg(global_model, client_models, ni, ns)
            test_loss, acc = utils.test(global_model, test_loader_batch)
            logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
            print('average train loss %0.6f | test loss %0.6f | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
        else:
            logging.info("Adversary selected in this round..")
            print("Adversary selected in this round..")
            if len(attacker_selected_rounds) != args.analysis_window:
                print("Analysing parameters for backdoor injection..")
                logging.info("Analysing parameters for backdoor injection..")
                attacker_selected_rounds.append(r)
                if utils.arch == "vgg11" or utils.arch == "vgg19":
                    perdoor_utils.extract_weights(global_model, r-1)
                else:
                    perdoor_utils.extract_weights_resnet(global_model, r-1)
                loss = 0
                for i in tqdm(range(utils.num_selected)):
                    loss += utils.client_update(client_models[i], opt[i], train_loader[client_idx[i]], client_idx[i])
                utils.fedavg(global_model, client_models, ni, ns)
                test_loss, acc = utils.test(global_model, test_loader_batch)
                logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
                print('average train loss %0.6f | test loss %0.6f | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
            else:
                torch.save(global_model, utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r-1}_{utils.dataset}.pth")
                print("Injecting backdoor..")
                logging.info("Injecting backdoor..")
                
                benign_clients = client_idx[client_idx != utils.attacker_id]
                loss = 0
                for i in tqdm(range(utils.num_selected - 1)):
                    loss += utils.client_update(client_models[i], opt[i], train_loader[benign_clients[i]], benign_clients[i])

                perdoor_utils.find_clean_images(global_model)
                clean_images = np.load(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"clean_images_source_{utils.source_class}_target_{utils.target_class}_{utils.dataset}.npy")
                perdoor_utils.create_backdoor_trigger(global_model, clean_images, args.eps)
                trigger = np.load(utils.image_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"trigger_source_{utils.source_class}_target_{utils.target_class}_{utils.dataset}.npy")
                param_no_change = perdoor_utils.get_param_no_change(attacker_selected_rounds, args.t_delta)
                if utils.arch == "vgg11" or utils.arch == "vgg19":
                    param_not_important = perdoor_utils.get_param_not_important(global_model)
                else:
                    param_not_important = perdoor_utils.get_param_not_important_resnet(global_model)
                backdoor_params = []
                for i in range(len(param_no_change)):
                    backdoor_params.append(np.logical_and(param_no_change[i], param_not_important[i]))
                backdoor_params = np.array(backdoor_params, dtype=object)
                np.save(f"backdoor_params_{utils.dataset}.npy", backdoor_params)
                
                if utils.arch == "vgg11" or utils.arch == "vgg19":
                    perdoor_utils.create_backdoor_network(global_model, clean_images, trigger, r-1, args.delta)
                else:
                    perdoor_utils.create_backdoor_network_resnet(global_model, clean_images, trigger, r-1, args.delta)
                print("Backdoor inserted..")
                logging.info("Backdoor inserted..")
                local_backdoor_model = torch.load(utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r-1}_local_backdoor_{utils.dataset}.pth")
                global_original_model = torch.load(utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r-1}_{utils.dataset}.pth")
                local_backdoor_model_dict = local_backdoor_model.state_dict()
                for k in local_backdoor_model_dict.keys():
                    local_backdoor_model_dict[k] = torch.add((ns/ni[utils.num_selected - 1])*(local_backdoor_model.state_dict()[k].float() - global_original_model.state_dict()[k].float()), global_original_model.state_dict()[k].float())
                local_backdoor_model.load_state_dict(local_backdoor_model_dict)
                client_models[utils.num_selected - 1] = local_backdoor_model
                utils.fedavg(global_model, client_models, ni, ns)
                torch.save(global_model, utils.model_path+"_"+utils.dataset+"_"+utils.arch+"/"+f"round_{r}_{utils.dataset}.pth")
                _, clean_acc = utils.test(global_model, test_loader_batch)
                fool_images = 0
                fool_total = 0
                test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
                for inputs, labels in test_loader:
                    if labels == utils.source_class:
                        inputs = inputs.cuda()
                        perturbed_inputs = inputs + torch.tensor(trigger).cuda()
                        perturbed_outputs = global_model(perturbed_inputs)
                        _, perturbed_predicted = perturbed_outputs.max(1)
                        fool_images += (perturbed_predicted == utils.target_class).sum().item()
                        fool_total += 1
                fooling_rate = float(fool_images)/float(fool_total)
                print(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Clean Accuracy: {np.round(clean_acc*100, 2)}, Global Backdoor Accuracy: {fooling_rate*100}')
                logging.info(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Clean Accuracy: {np.round(clean_acc*100, 2)}, Global Backdoor Accuracy: {fooling_rate*100}')
                backdoor_inserted = True
    else:
        loss = 0
        for i in tqdm(range(utils.num_selected)):
            loss += utils.client_update(client_models[i], opt[i], train_loader[client_idx[i]], client_idx[i])
        utils.fedavg(global_model, client_models, ni, ns)
        test_loss, acc = utils.test(global_model, test_loader_batch)
        logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
        fool_images = 0
        fool_total = 0
        for inputs, labels in test_loader:
            if labels == utils.source_class:
                inputs = inputs.cuda()
                perturbed_inputs = inputs + torch.tensor(trigger).cuda()
                perturbed_outputs = global_model(perturbed_inputs)
                _, perturbed_predicted = perturbed_outputs.max(1)
                fool_images += (perturbed_predicted == utils.target_class).sum().item()
                fool_total += 1
        fooling_rate = float(fool_images)/float(fool_total)
        print(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {fooling_rate*100}')
        logging.info(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {fooling_rate*100}')