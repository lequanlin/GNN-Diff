import pdb

import hydra.utils
import numpy as np
from .base_task import  BaseTask
from data.data_proc import load_data_link
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import datetime
from core.utils import *
import glob
import omegaconf
import json
import time

torch.manual_seed(42)

class LPTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(LPTask, self).__init__(config, **kwargs)

        ### Get data ###
        self.data, self.num_features = load_data_link(self.cfg)
        self.train_data = self.data.train_link_data.cuda()
        self.val_data = self.data.val_link_data.cuda()
        self.test_data = self.data.test_link_data.cuda()

    # override the abstract method in base_task.py
    def set_param_data(self):
        param_data = PData(self.cfg.param)
        self.model = param_data.get_model()
        self.train_layer = param_data.get_train_layer()
        return param_data

    def test_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input

        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        output_list = []

        with torch.no_grad():

            if model.model_name.lower() in ['mlp','gcn','sage','appnp','chebnet']:
                node_embeddings = model(self.test_data.x, self.test_data.edge_index)
            else:
                raise ValueError(f"Unsupported model: {args.model}")

            outputs = (node_embeddings[self.test_data.edge_label_index[0]] * node_embeddings[self.test_data.edge_label_index[1]]).sum(
                dim=-1).view(-1)
            test_loss = None
            pred = (outputs.sigmoid() > 0.5).cpu().numpy()
            acc = np.mean(pred == self.test_data.edge_label.cpu().numpy())
            output_list += pred.tolist()

        del model
        return acc, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input

        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        output_list = []

        with torch.no_grad():

            if model.model_name.lower() in ['mlp','gcn','sage','appnp','chebnet']:
                node_embeddings = model(self.val_data.x, self.val_data.edge_index)
            else:
                raise ValueError(f"Unsupported model: {args.model}")

            outputs = (node_embeddings[self.val_data.edge_label_index[0]] * node_embeddings[self.val_data.edge_label_index[1]]).sum(
                dim=-1).view(-1)
            test_loss = None
            pred = (outputs.sigmoid() > 0.5).cpu().numpy()
            acc = np.mean(pred == self.val_data.edge_label.cpu().numpy())
            output_list += pred.tolist()

        del model
        return acc, test_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):


        criterion = nn.BCEWithLogitsLoss()

        epoch = self.cfg.epoch # The epoch number at which we fix partial parameters
        save_num = self.cfg.save_num_model

        train_layer = self.cfg.train_layer

        ### Create saving path ###

        # "getattr": get the value of an attribute of an object, when the attribute does not exist, return the default (last input)
        data_path = getattr(self.cfg, 'save_root', 'param_data')
        data_path = os.path.join(data_path, 'link')
        data_path = os.path.join(data_path, self.cfg.model_name)

        tmp_path = os.path.join(data_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        save_model_accs = []
        save_model_test_accs = []
        parameters = []

        start_time = time.time()

        ### Training the target model to sample parameters ###
        net = self.build_model()  # The model is specified in folder "config/task"; the code of model is stored in folder "models"
        net = net.cuda()

        # for name, weights in net.named_parameters(): print(name)

        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]

        optimizer = self.build_optimizer_data_prep(self.cfg.optimizer1, net)

        scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)

        best_acc = 0
        test_acc = 0

        print('### Training for parameter collection starts ###')

        for i in range(0, epoch):

            ### Training ###
            train_acc = self.train(net, self.train_data, criterion, optimizer)

            ### Validation ###
            val_acc = self.test(net, self.val_data)
            best_acc = max(val_acc, best_acc)

            ### Testing ###
            test_acc_temp = self.test(net, self.test_data)

            if i <= (epoch - 1):

                ### Find the best validation model ###
                if best_acc == val_acc:
                    test_acc = test_acc_temp
                    torch.save(net, os.path.join(tmp_path, "whole_model.pth"))

            scheduler.step()

        test_acc_all = []
        for i in range(save_num):


            ### Fix partial parameters ###
            if (i+10) % 10 == 0:

                net = torch.load(os.path.join(tmp_path, "whole_model.pth"))
                optimizer = self.build_optimizer_data_prep(self.cfg.optimizer2, net)

                # Fix all parameters not in the train_layer list
                fix_partial_model(train_layer, net)
                parameters = []

                best_acc = 0

            ### Training ###
            train_acc = self.train(net, self.train_data, criterion, optimizer)

            ### Validation ###
            val_acc = self.test(net, self.val_data)
            best_acc = max(val_acc, best_acc)

            ### Testing ###
            test_acc = self.test(net, self.test_data)


            ### Record the test accuracy of each run ###
            if best_acc == val_acc:
                test_acc_temp = test_acc
            if i % 10 == 9:
                test_acc_all.append(test_acc_temp)


            ## Data collection of parameters ###
            parameters.append(state_part(train_layer, net))  # Get parameter samples
            save_model_accs.append(val_acc)
            save_model_test_accs.append(test_acc)


            if len(parameters) == 10 or i == save_num - 1:  # Save parameters and empty the parameter list in consideration of memory
                train_iter = i // 10 + 1
                train_epoch = i % 10
                torch.save(parameters, os.path.join(tmp_path, "p_data_{}_{}.pt".format(train_iter, train_epoch)))
                parameters = []


        end_time = time.time()
        print('Parameter collection finishes.')
        print('Search time cost (s):', (end_time - start_time))


        ### Load collected parameters and apply vectorization###
        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'performance': save_model_accs,
            'test_performance': save_model_test_accs,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_accs,
            'test_performance': save_model_test_accs,

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, train_data, criterion, optimizer):

        net.train()
        optimizer.zero_grad()
        if net.model_name.lower() in ['gcn', 'chebnet','appnp', 'sage', 'mlp']:
            node_embeddings = net(train_data.x, train_data.edge_index)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        # outputs are the product of a pair of nodes on each edge
        outputs = (node_embeddings[edge_label_index[0]] * node_embeddings[edge_label_index[1]]).sum(dim=-1).view(-1)

        loss = criterion(outputs, edge_label)
        loss.backward()
        optimizer.step()

        pred = (outputs > 0.5).cpu().numpy()
        train_acc = np.mean(pred == edge_label.cpu().numpy())

        return train_acc


    def test(self, net, data):

        global best_acc
        net.eval()

        with torch.no_grad():
            if net.model_name.lower() in ['gcn', 'chebnet', 'appnp', 'sage', 'mlp']:
                node_embeddings = net(data.x, data.edge_index)
            else:
                raise ValueError(f"Unsupported model: {args.model}")

            # outputs are the product of a pair of nodes on each edge
            outputs = (node_embeddings[data.edge_label_index[0]] * node_embeddings[data.edge_label_index[1]]).sum(
                dim=-1).view(-1).sigmoid()
            pred = (outputs > 0.5).cpu().numpy()
            eval_acc = np.mean(pred == data.edge_label.cpu().numpy())

        return eval_acc



