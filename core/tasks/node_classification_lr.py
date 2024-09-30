import pdb

import hydra.utils

from .base_task import  BaseTask
from data.data_proc import load_data_lrgb
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
from sklearn.metrics import f1_score
import datetime
from core.utils import *
import glob
import omegaconf
import json
import time

torch.manual_seed(42)

class NCFTask_LR(BaseTask):
    def __init__(self, config, **kwargs):
        super(NCFTask_LR, self).__init__(config, **kwargs)

        ### Get data ###
        if config.data.dataset.lower() in ['pascalvoc-sp', 'coco-sp']:
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = load_data_lrgb(self.cfg)
        else:
            raise ValueError(f"Unsupported dataset: {config.data.dataset}")

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

        test_loss = f1 = 0

        with torch.no_grad():

            for i, batch in enumerate(self.test_loader):
                batch = batch.cuda()
                if net.model_name.lower() in ['gcn', 'appnp', 'gin', 'sage', 'sgc', 'gprgnn', 'mixhop', 'mlp']:
                    outputs = model(batch.x, batch.edge_index)
                else:
                    raise ValueError(f"Unsupported model: {net.model_name}")

                predicted = outputs.max(1)[1].cpu().numpy()
                labels = batch.y.cpu().numpy()
                # Calculate loss
                test_loss_temp = self.weighted_cross_entropy(outputs, batch.y)
                test_loss += test_loss_temp
                # Calculate F1-score
                f1_temp = f1_score(labels, predicted, average='macro', zero_division=0)
                f1 += f1_temp
                output_list.extend(outputs)

        test_loss /= len(self.test_loader)
        f1 /= len(self.test_loader)

        del model
        return f1, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input

        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        output_list = []

        test_loss = f1 = 0

        with torch.no_grad():

            for i, batch in enumerate(self.val_loader):
                batch = batch.cuda()
                if net.model_name.lower() in ['gcn', 'appnp', 'gin', 'sage', 'sgc', 'gprgnn', 'mixhop', 'mlp']:
                    outputs = model(batch.x, batch.edge_index)
                else:
                    raise ValueError(f"Unsupported model: {net.model_name}")

                predicted = outputs.max(1)[1].cpu().numpy()
                labels = batch.y.cpu().numpy()
                # Calculate loss
                test_loss_temp = self.weighted_cross_entropy(outputs, batch.y)
                test_loss += test_loss_temp
                # Calculate F1-score
                f1_temp = f1_score(labels, predicted, average='macro', zero_division=0)
                f1 += f1_temp
                output_list.extend(outputs)

        test_loss /= len(self.val_loader)
        f1 /= len(self.val_loader)

        del model
        return f1, test_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):

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

        save_model_f1s = []
        save_model_test_f1s = []
        parameters = []

        start_time = time.time()

        ### Training the target model to sample parameters ###
        net = self.build_model()  # The model is specified in folder "config/task"; the code of model is stored in folder "models"
        net = net.cuda()

        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]

        optimizer = self.build_optimizer_data_prep(self.cfg.optimizer1, net)

        best_f1 = 0
        test_f1 = 0

        print('### Training for parameter collection starts ###')

        for i in range(0, epoch):

            train_f1 = val_f1 = current_test_f1 = 0

            ### Training ###
            for j, batch_train in enumerate(self.train_loader):
                batch_train = batch_train.cuda()
                batch_train.edge_index = to_undirected(batch_train.edge_index, num_nodes=batch_train.x.shape[0])
                train_f1_temp = self.train(net, optimizer, batch_train) #We train with 50 batches in each epoch
                train_f1 += train_f1_temp
                if j == 49: break
            train_f1 /= 50

            ### Validation and Testing ###
            for j, batch_val in enumerate(self.val_loader):
                batch_val = batch_val.cuda()
                batch_val.edge_index = to_undirected(batch_val.edge_index, num_nodes=batch_val.x.shape[0])
                val_f1_temp = self.eval_f1(net, batch_val)
                val_f1 += val_f1_temp
            val_f1 /= len(self.val_loader)

            best_f1 = max(val_f1, best_f1)

            if i <= (epoch - 1):

                ### Find the best validation model ###
                if best_f1 == val_f1:
                    torch.save(net, os.path.join(tmp_path, "whole_model.pth"))


            # scheduler.step()

        test_f1_all = []
        for i in range(save_num):

            ### Fix partial parameters ###
            if (i+10) % 10 == 0:

                net = torch.load(os.path.join(tmp_path, "whole_model.pth"))
                optimizer = self.build_optimizer_data_prep(self.cfg.optimizer2, net)


                # Fix all parameters not in the train_layer list
                fix_partial_model(train_layer, net)
                parameters = []

                best_f1 = 0

            train_f1 = val_f1 = current_test_f1 = 0

            ### Training ###
            for j, batch_train in enumerate(self.train_loader):
                batch_train = batch_train.cuda()
                batch_train.edge_index = to_undirected(batch_train.edge_index, num_nodes=batch_train.x.shape[0])
                train_f1_temp = self.train(net, optimizer, batch_train) #We train with 10 batches in each epoch
                train_f1 += train_f1_temp
                if j == 9: break
            train_f1 /= 10

            ### Validation and Testing ###
            for j, batch_val in enumerate(self.val_loader):
                batch_val = batch_val.cuda()
                batch_val.edge_index = to_undirected(batch_val.edge_index, num_nodes=batch_val.x.shape[0])
                val_f1_temp = self.eval_f1(net, batch_val)
                val_f1 += val_f1_temp
            val_f1 /= len(self.val_loader)

            best_f1 = max(val_f1, best_f1)


            ### Record the test f1 of each run ###
            if best_f1 == val_f1:
                for j, batch_test in enumerate(self.test_loader):
                    batch_test = batch_test.cuda()
                    batch_test.edge_index = to_undirected(batch_test.edge_index, num_nodes=batch_test.x.shape[0])
                    test_f1_temp = self.eval_f1(net, batch_test)
                    current_test_f1 += test_f1_temp
                current_test_f1 /= len(self.test_loader)
                test_f1 = current_test_f1
            if i % 10 == 9:
                test_f1_all.append(test_f1)

            ## Data collection of parameters ###
            parameters.append(state_part(train_layer, net))  # Get parameter samples
            save_model_f1s.append(val_f1)
            save_model_test_f1s.append(test_f1)

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
            'performance': save_model_f1s,
            'test_performance': save_model_test_f1s,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_f1s,
            'test_performance': save_model_test_f1s,

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, optimizer, data):

        net.train()
        optimizer.zero_grad()
        if net.model_name.lower() in ['gcn', 'appnp', 'gin', 'sage', 'sgc', 'gprgnn', 'mixhop', 'mlp']:
            logits = net(data.x, data.edge_index)
        else:
            raise ValueError(f"Unsupported model: {net.model_name}")

        loss = self.weighted_cross_entropy(logits,data.y)
        loss.backward()
        optimizer.step()

        pred = logits.max(1)[1].cpu().numpy()
        labels = data.y.cpu().numpy()

        # Calculate F1-score
        train_f1 = f1_score(labels, pred, average='macro', zero_division=0)

        return train_f1
    def eval_f1(self, net, data):
        global best_f1
        net.eval()
        if net.model_name.lower() in ['gcn', 'appnp', 'gin', 'sage', 'sgc', 'gprgnn', 'mixhop', 'mlp']:
            logits = net(data.x, data.edge_index)
        else:
            raise ValueError(f"Unsupported model: {net.model_name}")

        pred = logits.max(1)[1].cpu().numpy()
        labels = data.y.cpu().numpy()

        # Calculate F1-score
        f1 = f1_score(labels, pred, average='macro', zero_division=0)

        return f1

    def weighted_cross_entropy(self, pred, true):
        """
        Weighted cross-entropy for unbalanced classes.
        """
        # calculating label weights for weighted loss computation
        num_nodes = true.size(0)
        num_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(num_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (num_nodes - cluster_sizes).float() / num_nodes
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight)
        # binary: in case we only have two unique classes in y
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
        return loss




