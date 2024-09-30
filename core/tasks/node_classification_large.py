import pdb

import hydra.utils

from .base_task import  BaseTask
from data.data_proc import load_data_ogb, load_data_pyg_large
from torch_geometric.loader import ClusterData, ClusterLoader
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

class NCFTask_Large(BaseTask):
    def __init__(self, config, **kwargs):
        super(NCFTask_Large, self).__init__(config, **kwargs)

        ### Get data ###
        if config.data.dataset.lower() in ['ogbn-products']:
            self.data, self.num_features, self.num_classes = load_data_ogb(self.cfg)
            cluster_num_parts = 960
        elif config.data.dataset.lower() in ['ogbn-arxiv']:
            self.data, self.num_features, self.num_classes = load_data_ogb(self.cfg)
            cluster_num_parts = 32
        elif config.data.dataset.lower() in ['flickr']:
            self.data, self.num_features, self.num_classes = load_data_pyg_large(self.cfg)
            cluster_num_parts = 32
        elif config.data.dataset.lower() in ['reddit']:
            self.data, self.num_features, self.num_classes = load_data_pyg_large(self.cfg)
            cluster_num_parts = 960
        else:
            raise ValueError(f"Unsupported dataset: {config.data.dataset}")

        ### Get Loader ###
        # Create clusters from the large graph
        cluster_data = ClusterData(self.data, num_parts=cluster_num_parts, recursive=False)
        # Create a data loader for the clusters
        self.train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True, num_workers=0)
        self.eval_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=False, num_workers=0)

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

            test_acc = test_loss = 0

            for i, batch in enumerate(self.eval_loader):
                batch = batch.cuda()
                test_acc_temp, loss_temp, output_list_temp = self.eval(model, batch, batch.test_mask)
                test_acc += test_acc_temp
                test_loss += loss_temp
                output_list.extend(output_list_temp)
            num_batchs = len(self.eval_loader)
            test_acc /= num_batchs
            test_loss /= num_batchs

        del model
        return test_acc, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input

        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        output_list = []

        with torch.no_grad():

            val_acc = val_loss = 0

            for i, batch in enumerate(self.eval_loader):
                batch = batch.cuda()
                val_acc_temp, loss_temp, output_list_temp = self.eval(model, batch, batch.val_mask)
                val_acc += val_acc_temp
                val_loss += loss_temp
                output_list.extend(output_list_temp)
            num_batchs = len(self.eval_loader)
            val_acc /= num_batchs
            val_loss /= num_batchs

        del model
        return val_acc, val_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):


        criterion = nn.CrossEntropyLoss()

        epoch = self.cfg.epoch # The epoch number at which we fix partial parameters
        save_num = self.cfg.save_num_model

        train_layer = self.cfg.train_layer

        ### Create saving path ###

        # "getattr": get the value of an attribute of an object, when the attribute does not exist, return the default (last input)
        data_path = getattr(self.cfg, 'save_root', 'param_data')
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

            train_acc = val_acc = current_test_acc = 0

            ### Training ###
            for j, batch in enumerate(self.train_loader):
                batch = batch.cuda()
                train_acc_temp = self.train(net, criterion, optimizer, i, batch, batch.train_mask)
                train_acc += train_acc_temp

            num_batchs = len(self.train_loader)
            train_acc = train_acc / num_batchs


            ### Validation and Testing###
            for j, batch in enumerate(self.eval_loader):
                batch = batch.cuda()
                val_acc_temp = self.test(net, batch, batch.val_mask)
                test_acc_temp = self.test(net, batch, batch.test_mask)
                val_acc += val_acc_temp
                current_test_acc += test_acc_temp

            num_batchs = len(self.eval_loader)
            val_acc = val_acc / num_batchs
            current_test_acc = current_test_acc / num_batchs

            best_acc = max(val_acc, best_acc)

            if i <= (epoch - 1):

                ### Find the best validation model ###
                if best_acc == val_acc:
                    test_acc = current_test_acc
                    torch.save(net, os.path.join(tmp_path, "whole_model.pth"))


            # scheduler.step()

        test_acc_all = []
        for i in range(save_num):

            ### Fix partial parameters ###
            if (i+10) % 10 == 0:

                net = torch.load(os.path.join(tmp_path, "whole_model.pth"))
                optimizer = self.build_optimizer_data_prep(self.cfg.optimizer2, net)
                val_acc = current_test_acc = 0
                for j, batch in enumerate(self.eval_loader):
                    batch = batch.cuda()
                    val_acc_temp = self.test(net, batch, batch.val_mask)
                    test_acc_temp = self.test(net, batch, batch.test_mask)
                    val_acc += val_acc_temp
                    current_test_acc += test_acc_temp

                num_batchs = len(self.eval_loader)
                val_acc_fixed = val_acc / num_batchs
                test_acc_fixed = current_test_acc / num_batchs
                print('Fixed model: validation accuracy {}, test accuracy {}'.format(val_acc_fixed, test_acc_fixed))

                # Fix all parameters not in the train_layer list
                fix_partial_model(train_layer, net)
                parameters = []

                best_acc = 0

            train_acc = val_acc = current_test_acc = 0

            ### Training ###
            for j, batch in enumerate(self.train_loader):
                batch = batch.cuda()
                train_acc_temp = self.train(net, criterion, optimizer, i, batch, batch.train_mask)
                train_acc += train_acc_temp

            num_batchs = len(self.train_loader)
            train_acc = train_acc / num_batchs

            ### Validation and Testing###
            for j, batch in enumerate(self.eval_loader):
                batch = batch.cuda()
                val_acc_temp = self.test(net, batch, batch.val_mask)
                test_acc_temp = self.test(net, batch, batch.test_mask)
                val_acc += val_acc_temp
                current_test_acc += test_acc_temp

            num_batchs = len(self.eval_loader)
            val_acc = val_acc / num_batchs
            current_test_acc = current_test_acc / num_batchs

            best_acc = max(val_acc, best_acc)

            ### Record the test accuracy of each run ###
            if best_acc == val_acc:
                test_acc = current_test_acc
            if i % 10 == 9:
                test_acc_all.append(test_acc)


            ## Data collection of parameters ###
            parameters.append(state_part(train_layer, net))  # Get parameter samples
            save_model_accs.append(val_acc)
            save_model_test_accs.append(current_test_acc)

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

    def train(self, net, criterion, optimizer, epoch, data, mask):

        net.train()
        optimizer.zero_grad()
        if net.model_name.lower() in ['gcn', 'appnp', 'sage', 'gpr']:
            outputs = net(data.x, data.edge_index)
        else:
            raise ValueError(f"Unsupported model: {net.model_name}")

        loss = criterion(outputs[mask],data.y[mask])
        loss.backward()
        optimizer.step()

        _, predicted = outputs[mask].max(1)
        total = mask.sum().item()
        correct = predicted.eq(data.y[mask]).sum().item()
        train_acc = 100. * correct / total

        return train_acc


    def test(self, net, data, mask):

        global best_acc
        net.eval()

        with torch.no_grad():
            if net.model_name.lower() in ['gcn', 'appnp', 'sage', 'gpr']:
                outputs = net(data.x, data.edge_index)
            else:
                raise ValueError(f"Unsupported model: {net.model_name}")

            _, predicted = outputs[mask].max(1)
            total = mask.sum().item()
            correct = predicted.eq(data.y[mask]).sum().item()
            eval_acc = 100. * correct / total

        return eval_acc

    def eval(self, net, data, mask):
        output_list = []

        global best_acc
        # net.eval()

        with torch.no_grad():
            if net.model_name.lower() in ['gcn', 'appnp', 'sage', 'gpr']:
                outputs = net(data.x, data.edge_index)
            else:
                raise ValueError(f"Unsupported model: {net.model_name}")

            _, predicted = outputs[mask].max(1)
            total = mask.sum().item()
            test_loss = F.cross_entropy(outputs[mask], data.y[mask], reduction='mean').item()
            correct = predicted.eq(data.y[mask]).sum().item()
            eval_acc = 100. * correct / total
            output_list += predicted.cpu().numpy().tolist()

        return eval_acc, test_loss, output_list



