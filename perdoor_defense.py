import torch
import torch.optim as optim

import argparse
import logging
from tqdm import tqdm

import numpy as np
import os

import utils
import perdoor_utils
import dataLoader

from vgg_model import VGG

# Use this manual seed to reproduce all results
# ==================================================
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument('--method', action='store', type=str, required=True)
parser.add_argument('--analysis_window', action='store', type=int, required=True)
parser.add_argument('--t_delta', action='store', type=float, required=True)
parser.add_argument('--delta', action='store', type=float, required=True)
parser.add_argument('--eps', action='store', type=float, required=True)
args = parser.parse_args()

logging.basicConfig(filename=f'fl_training_log_{args.method}.log', level=logging.INFO, format='%(message)s')
if not os.path.exists(utils.model_path):
    os.mkdir(utils.model_path)
global_model =  VGG('VGG11').cuda()
torch.save(global_model, utils.model_path+f"round_0_global.pth")
client_models = [VGG('VGG11').cuda() for _ in range(utils.num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]
attacker_selected_rounds = []
backdoor_inserted = False
for r in range(1, utils.num_rounds):
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
                loss += utils.client_update(client_models[i], opt[i], dataLoader.train_loader[client_idx[i]], client_idx[i])
            if args.method == "krum":
                utils.krum(global_model, client_models, r)
            if args.method == "foolsgold":
                utils.foolsgold(global_model, client_models, r)
            test_loss, acc = utils.test(global_model, dataLoader.test_loader)
            logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
            print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
        else:
            logging.info("Adversary selected in this round..")
            print("Adversary selected in this round..")
            if len(attacker_selected_rounds) != args.analysis_window:
                print("Analysing parameters for backdoor injection..")
                logging.info("Analysing parameters for backdoor injection..")
                attacker_selected_rounds.append(r)
                perdoor_utils.extract_weights(global_model, r-1)
                loss = 0
                for i in tqdm(range(utils.num_selected)):
                    loss += utils.client_update(client_models[i], opt[i], dataLoader.train_loader[client_idx[i]], client_idx[i])
                if args.method == "krum":
                    utils.krum(global_model, client_models, r)
                if args.method == "foolsgold":
                    utils.foolsgold(global_model, client_models, r)
                test_loss, acc = utils.test(global_model, dataLoader.test_loader)
                logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
                print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
            else:
                print("Injecting backdoor..")
                logging.info("Injecting backdoor..")
                benign_clients = client_idx[client_idx != utils.attacker_id]
                loss = 0
                for i in tqdm(range(utils.num_selected - 1)):
                    loss += utils.client_update(client_models[i], opt[i], dataLoader.train_loader[benign_clients[i]], benign_clients[i])
                param_no_change = perdoor_utils.get_param_no_change(attacker_selected_rounds, args.t_delta)
                param_not_important = perdoor_utils.get_param_not_important(global_model, r-1)
                backdoor_params = []
                for i in range(len(param_no_change)):
                    backdoor_params.append(np.logical_and(param_no_change[i], param_not_important[i]))
                np.save("backdoor_params.npy", backdoor_params)
                to_be_backdoor_images = np.load(utils.backdoor_image_path+"images.npy")
                backdoor_images = perdoor_utils.create_non_uniform_backdoor_images(global_model, to_be_backdoor_images, args.eps)
                
                model = torch.load(utils.model_path+f"round_{r-1}_global.pth")
                _, acc = utils.test(model, dataLoader.test_loader)
                backdoor_correct = 0
                for item in backdoor_images:
                    item = torch.tensor(item).cuda()
                    output = model(item)
                    if output.argmax().item() == utils.target_class:
                        backdoor_correct += 1
                print(f'Source: {utils.source_class}, Target: {utils.target_class}, Adversarial Backdoor Accuracy: {backdoor_correct}, Clean Accuracy: {np.round(acc*100, 2)}')
                
                model = torch.load(utils.model_path+f"round_{r-1}_global.pth")
                perdoor_utils.create_backdoor_network(model, backdoor_images, r-1, args.delta, utils.target_class)
                print("Backdoor inserted..")
                logging.info("Backdoor inserted..")
                model = torch.load(utils.model_path+f"round_{r-1}_local_backdoor_target_{utils.target_class}.pth")
                model.eval()
                _, acc = utils.test(model, dataLoader.test_loader)
                backdoor_correct = 0
                for item in backdoor_images:
                    item = torch.tensor(item).cuda()
                    output = model(item)
                    if output.argmax().item() == utils.target_class:
                        backdoor_correct += 1
                print(f'Source: {utils.source_class}, Target: {utils.target_class}, Local Backdoor Accuracy: {backdoor_correct}, Local Clean Accuracy: {np.round(acc*100, 2)}')
                logging.info(f'Source: {utils.source_class}, Target: {utils.target_class}, Local Backdoor Accuracy: {backdoor_correct}, Local Clean Accuracy: {np.round(acc*100, 2)}')
                local_backdoor_model = torch.load(utils.model_path+f"round_{r-1}_local_backdoor_target_{utils.target_class}.pth")
                client_models[utils.num_selected - 1] = local_backdoor_model
                if args.method == "krum":
                    utils.krum(global_model, client_models, r)
                if args.method == "foolsgold":
                    utils.foolsgold(global_model, client_models, r)
                model = torch.load(utils.model_path+f"round_{r}_global.pth")
                model.eval()
                _, clean_acc = utils.test(model, dataLoader.test_loader)
                backdoor_correct = 0
                for item in backdoor_images:
                    item = torch.tensor(item).cuda()
                    output = model(item)
                    if output.argmax().item() == utils.target_class:
                        backdoor_correct += 1
                print(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {backdoor_correct}, Global Clean Accuracy: {np.round(clean_acc*100, 2)}')
                logging.info(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {backdoor_correct}, Global Clean Accuracy: {np.round(clean_acc*100, 2)}')
                backdoor_inserted = True
    else:
        loss = 0
        for i in tqdm(range(utils.num_selected)):
            loss += utils.client_update(client_models[i], opt[i], dataLoader.train_loader[client_idx[i]], client_idx[i])
        if args.method == "krum":
            utils.krum(global_model, client_models, r)
        if args.method == "foolsgold":
            utils.foolsgold(global_model, client_models, r)
        test_loss, acc = utils.test(global_model, dataLoader.test_loader)
        logging.info(f"Global Model Loss: {test_loss}, Accuracy: {acc}")
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / utils.num_selected, test_loss, acc))
        backdoor_images = np.load(utils.backdoor_image_path+f"backdoor_images.npy")
        model = torch.load(utils.model_path+f"round_{r}_global.pth")
        model.eval()
        backdoor_correct = 0
        for item in backdoor_images:
            item = torch.tensor(item).cuda()
            output = model(item)
            if output.argmax().item() == utils.target_class:
                backdoor_correct += 1
        print(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {backdoor_correct}')
        logging.info(f'Source: {utils.source_class}, Target: {utils.target_class}, Global Backdoor Accuracy: {backdoor_correct}')