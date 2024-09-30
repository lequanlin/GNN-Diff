import pdb

import hydra.utils

from .base_task import  BaseTask
from data.data_proc import load_data, generate_split
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import datetime
from core.utils import *
import glob
import omegaconf
import json
import time

torch.manual_seed(42)

class NCFTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(NCFTask, self).__init__(config, **kwargs)

        ### Get data ###
        self.data, self.num_features, self.num_classes = load_data(self.cfg)
        if config.data.dataset.lower() in ['texas', 'cornell', 'wisconsin', 'actor','chameleon']:
            # Use the first split to try
            self.train_mask = self.data.train_mask[:, 0]
            self.val_mask = self.data.val_mask[:, 0]
            self.test_mask = self.data.test_mask[:, 0]
        elif config.data.dataset.lower() in ['computers', 'photo', 'cs', 'physics']:
            self.train_mask, self.val_mask, self.test_mask = generate_split(self.data, self.num_classes)
        elif config.data.dataset.lower() in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
            self.train_mask, self.val_mask, self.test_mask = self.data.stores[0]['train_mask'][:, 0], self.data.stores[0]['val_mask'][:, 0], self.data.stores[0]['test_mask'][:, 0]
        elif config.data.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
            self.train_mask = self.data.train_mask
            self.val_mask = self.data.val_mask
            self.test_mask = self.data.test_mask

        edge_index, edge_weight = gcn_norm(self.data.edge_index, num_nodes=self.data.x.shape[0], add_self_loops=True)
        self.data.sparse_adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                 (self.data.x.shape[0], self.data.x.shape[0])).cuda()
        self.data.adj_gt = torch.sparse.FloatTensor(edge_index, torch.ones_like(edge_weight),
                                                    (self.data.x.shape[0], self.data.x.shape[0])).to_dense().cuda()

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
        data = self.data.cuda()
        output_list = []

        with torch.no_grad():

            if model.model_name in ['MLP', 'GCN','GAT','ChebNet','APPNP','GIN','SAGE','SGC','GPRGNN','MixHop']:
                outputs = model(data.x, data.edge_index)
            elif model.model_name in ['H2']:
                outputs = model(data.x, data.sparse_adj)

            _, predicted = outputs[self.test_mask].max(1)
            total = self.test_mask.sum().item()
            test_loss = F.cross_entropy(outputs[self.test_mask], data.y[self.test_mask], reduction='sum').item()
            correct = predicted.eq(data.y[self.test_mask]).sum().item()
            test_loss /= total
            acc = 100. * correct / total
            output_list += predicted.cpu().numpy().tolist()

        del model
        return acc, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input

        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        data = self.data.cuda()
        output_list = []

        with torch.no_grad():

            if model.model_name in ['MLP', 'GCN','GAT','ChebNet','APPNP','GIN', 'SAGE','SGC','GPRGNN','MixHop']:
                outputs = model(data.x, data.edge_index)
            elif model.model_name in ['H2']:
                outputs = model(data.x, data.sparse_adj)

            _, predicted = outputs[self.val_mask].max(1)
            total = self.val_mask.sum().item()
            test_loss = F.cross_entropy(outputs[self.val_mask], data.y[self.val_mask], reduction='sum').item()
            correct = predicted.eq(data.y[self.val_mask]).sum().item()
            test_loss /= total
            acc = 100. * correct / total
            output_list += predicted.cpu().numpy().tolist()

        del model
        return acc, test_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):


        criterion = nn.CrossEntropyLoss()

        epoch = self.cfg.epoch # The epoch number at which we fix partial parameters
        save_num = self.cfg.save_num_model
        all_epoch = epoch + save_num


        train_layer = self.cfg.train_layer

        ### Create saving path ###

        # "getattr": get the value of an attribute of an object, when the attribute does not exist, return the default (last input)
        data_path = getattr(self.cfg, 'save_root', 'param_data')
        data_path = os.path.join(data_path, self.cfg.model_name)

        tmp_path = os.path.join(data_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        # tmp_path = os.path.join(data_path, 'tmp')
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        best_test_acc = 0
        best_val_acc = 0
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
            train_acc = self.train(net, criterion, optimizer, i, self.train_mask)

            ### Validation ###
            val_acc = self.test(net, self.val_mask)
            best_acc = max(val_acc, best_acc)

            ### Testing ###
            test_acc_temp = self.test(net, self.test_mask)

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
            train_acc = self.train(net, criterion, optimizer, i, self.train_mask)

            ### Validation ###
            val_acc = self.test(net, self.val_mask)
            best_acc = max(val_acc, best_acc)

            ### Testing ###
            test_acc = self.test(net, self.test_mask)


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
        return {'save_path': final_path}

    def train(self, net, criterion, optimizer, epoch, mask):

        data = self.data.cuda()

        net.train()
        optimizer.zero_grad()
        if net.model_name in ['MLP','GCN','GAT','ChebNet','APPNP','GIN','SAGE','SGC','GPRGNN','MixHop']:
            outputs = net(data.x, data.edge_index)
        elif net.model_name in ['H2']:
            outputs = net(data.x, data.sparse_adj)

        loss = criterion(outputs[mask],data.y[mask])
        loss.backward()
        optimizer.step()

        _, predicted = outputs[mask].max(1)
        total = mask.sum().item()
        correct = predicted.eq(data.y[mask]).sum().item()
        train_acc = 100. * correct / total

        return train_acc


    def test(self, net, mask):

        global best_acc
        data = self.data.cuda()
        net.eval()

        with torch.no_grad():
            # outputs = net(data.x,data.edge_index)
            if net.model_name in ['MLP','GCN','GAT','ChebNet','APPNP','GIN','SAGE','SGC','GPRGNN','MixHop']:
                outputs = net(data.x, data.edge_index)
            elif net.model_name in ['H2']:
                outputs = net(data.x, data.sparse_adj)
            _, predicted = outputs[mask].max(1)
            total = mask.sum().item()
            correct = predicted.eq(data.y[mask]).sum().item()
            eval_acc = 100. * correct / total

        return eval_acc



