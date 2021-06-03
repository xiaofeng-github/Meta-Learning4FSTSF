# -*- coding:utf-8 -*-
__author__ = 'XF'

'meta network'
import os
import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
from tools import metrics as Metrics

torch.set_default_tensor_type(torch.DoubleTensor)


class BaseCNN(nn.Module):

    def __init__(self, output=10):
        super(BaseCNN, self).__init__()

        self.output = output

        # this list contains all tensor needed to be optimized
        self.vars = nn.ParameterList()

        # running_mean and running var
        self.vars_bn = nn.ParameterList()

        # Conv1d layer
        # [channel_out, channel_in, kernel-size]
        weight = nn.Parameter(torch.ones(64, 1, 3))

        nn.init.kaiming_normal_(weight)

        bias = nn.Parameter(torch.zeros(64))

        self.vars.extend([weight, bias])

        # linear layer
        weight = nn.Parameter(torch.ones(self.output, 64 * 100))
        bias = nn.Parameter(torch.zeros(self.output))

        self.vars.extend([weight, bias])

    def forward(self, x, vars=None, bn_training=True):
        '''

        :param x: [batch size, 1, 3, 94]
        :param vars:
        :param bn_training: set false to not update
        :return:
        '''

        if vars is None:
            vars = self.vars

        x = x.squeeze(dim=2)
        x = x.unsqueeze(dim=1)
        # Conv1d layer
        weight, bias = vars[0].to(x.device), vars[1].to(x.device)
        # x ==> (batch size, 1, 200)

        x = F.conv1d(x, weight, bias, stride=1, padding=1)  # ==>(batch size, 64, 200)
        x = F.relu(x, inplace=True)  # ==> (batch_size, 64, 200)
        x = F.max_pool1d(x, kernel_size=2)  # ==> (batch_size, 64, 100)

        # linear layer
        x = x.view(x.size(0), -1)  # flatten ==> (batch_size, 16*12)
        weight, bias = vars[-2].to(x.device), vars[-1].to(x.device)
        x = F.linear(x, weight, bias)

        return x

    def parameters(self):
        return self.vars

    def zero_grad(self):
        pass

    pass

class BaseLSTM(nn.Module):

    def __init__(self, n_features, n_hidden, n_output, n_layer=1):
        super().__init__()
        self.name = 'BaseLSTM'

        # this list contains all tensor needed to be optimized
        self.params = nn.ParameterList()

        self.input_size = n_features
        self.hidden_size = n_hidden
        self.output_size = n_output
        self.layer_size = n_layer

        # input layer
        W_i = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.input_size))
        bias_i = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.params.extend([W_i, bias_i])

        # hidden layer
        W_h = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
        bias_h = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.params.extend([W_h, bias_h])

        if self.layer_size > 1:
            for _ in range(self.layer_size - 1):
                # i-th layer
                # input layer
                W_i = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
                bias_i = nn.Parameter(torch.Tensor(self.hidden_size * 4))
                self.params.extend([W_i, bias_i])
                # hidden layer
                W_h = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
                bias_h = nn.Parameter(torch.Tensor(self.hidden_size * 4))
                self.params.extend([W_h, bias_h])

        # output layer
        W_linear = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        bias_linear = nn.Parameter(torch.Tensor(self.output_size))
        self.params.extend([W_linear, bias_linear])

        self.init()

    pass

    def parameters(self):
        return self.params

    def init(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, vars=None, init_state=None):

        if vars is None:
            params = self.params
        else:
            params = vars

        # assume the shape of x is (batch_size, time_size, feature_size)
        batch_size, time_size, _ = x.size()
        hidden_seq = []

        if init_state is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_state

        HS = self.hidden_size

        for t in range(time_size):
            x_t = x[:, t, :]
            W_i, bias_i = (params[0].to(x.device), params[1].to(x.device))
            W_h, bias_h = (params[2].to(x.device), params[3].to(x.device))

            gates = F.linear(x_t, W_i, bias_i) + F.linear(h_t, W_h, bias_h)

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:])  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)

        W_linear, bias_linear = (params[-2].to(x.device), params[-1].to(x.device))
        out = F.linear(hidden_seq[-1], W_linear, bias_linear)
        return out


class BaseCNNConLSTM(nn.Module):

    def __init__(self, n_features, n_hidden, n_output, n_layer=1, time_size=1, cnn_feature=200):
        super(BaseCNNConLSTM, self).__init__()
        self.name = 'BaseCNNConLSTM'
        self.time_size = time_size

        # this list contain all tensor needed to be optimized
        self.params = nn.ParameterList()
        self.cnn = BaseCNN(output=cnn_feature)
        self.lstm = BaseLSTM(n_features=n_features, n_hidden=n_hidden, n_output=n_output, n_layer=n_layer)
        self.cnn_tensor_num = 0
        self.lstm_tensor_num = 0
        self.init()

    def init(self):

        self.cnn_tensor_num = len(self.cnn.parameters())
        self.lstm_tensor_num = len(self.lstm.parameters())
        for param in self.cnn.parameters():
            self.params.append(param)
        for param in self.lstm.parameters():
            self.params.append(param)

    def sequence(self, data):

        dim_1, dim_2 = data.shape
        new_dim_1 = dim_1 - self.time_size + 1

        x = torch.zeros((new_dim_1, self.time_size, dim_2))

        for i in range(dim_1 - self.time_size + 1):
            x[i] = data[i: i + self.time_size]
        return x.to(data.device)

    def forward(self, x, vars=None, init_states=None):

        if vars is None:
            params = self.params
        else:
            params = vars

        x = self.cnn(x, params[: self.cnn_tensor_num])
        # x = self.sequence(x)
        x = x.unsqueeze(dim=2)
        output = self.lstm(x, params[self.cnn_tensor_num:], init_states)
        return output
        pass

    pass


class MetaNet(nn.Module):

    def __init__(self, baseNet=None, update_step=10, update_step_test=20, meta_lr=0.001, base_lr=0.01, fine_lr=0.01):

        super(MetaNet, self).__init__()
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.meta_lr = meta_lr
        self.base_lr = base_lr
        self.fine_tune_lr = fine_lr

        if baseNet is not None:
            self.net = baseNet
        else:
            raise Exception('baseNet is None')
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        pass

    def forward(self, spt_x, spt_y, qry_x, qry_y, device='cpu'):
        '''

        :param spt_x: if baseNet is cnn: [ spt size, in_channel, height, width], lstm [spt_size, time_size, feature_size]
        :param spt_y: [ spt size]
        :param qry_x: if baseNet is cnn: [ qry size, in_channel, height, width], lstm [qry size, time_size, feature_size]
        :param qry_y: [ qry size]
        :return:

        '''

        task_num = len(spt_x)
        loss_list_qry = []
        rmse_list = []
        smape_list = []
        qry_loss_sum = 0
        # 0-th step update
        for i in range(task_num):
            x_spt = torch.from_numpy(spt_x[i]).to(device)
            y_spt = torch.from_numpy(spt_y[i]).to(device)
            x_qry = torch.from_numpy(qry_x[i]).to(device)
            y_qry = torch.from_numpy(qry_y[i]).to(device)

            y_hat = self.net(x_spt, vars=None)
            loss = F.mse_loss(y_hat, y_spt)
            grad = torch.autograd.grad(loss, self.net.parameters())
            grads_params = zip(grad, self.net.parameters())

            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], grads_params))

            with torch.no_grad():
                y_hat = self.net(x_qry, fast_weights)
                loss_qry = F.mse_loss(y_hat, y_qry)
                loss_list_qry.append(loss_qry)

                # calculating metrics
                rmse, smape = Metrics(y_qry, y_hat)

                rmse_list.append(rmse)
                smape_list.append(smape)

            for step in range(1, self.update_step):
                y_hat = self.net(x_spt, fast_weights)
                loss = F.mse_loss(y_hat, y_spt)
                grad = torch.autograd.grad(loss, fast_weights)
                grads_params = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], grads_params))

                if step < self.update_step - 1:
                    with torch.no_grad():
                        y_hat = self.net(x_qry, fast_weights)
                        loss_qry = F.mse_loss(y_hat, y_qry)
                        loss_list_qry.append(loss_qry)
                else:
                    y_hat = self.net(x_qry, fast_weights)
                    loss_qry = F.mse_loss(y_hat, y_qry)
                    loss_list_qry.append(loss_qry)
                    qry_loss_sum += loss_qry

                with torch.no_grad():
                    rmse, smape = Metrics(y_qry, y_hat)
                    rmse_list.append(rmse)
                    smape_list.append(smape)
                    pass

        # update meta net
        loss_qry = qry_loss_sum / task_num
        self.meta_optim.zero_grad()
        loss_qry.backward()
        self.meta_optim.step()

        return {
            'loss': loss_list_qry[-1].item(),
            'rmse': rmse_list[-1],
            'smape': smape_list[-1]
        }

    def fine_tuning(self, spt_x, spt_y, qry_x, qry_y, naive=False):

        '''

        :param spt_x: if baseNet is cnn:[set size, channel, height, width] if baseNet is lstm: [batch_size, seq_size, feature_size]
        :param spt_y:
        :param qry_x:
        :param qry_y:
        :return:
        '''

        # metrics
        loss_qry_list = []
        rmse_list = []
        smape_list = []
        min_train_loss = 1000000
        loss_set = {
            'train_loss': [],
            'validation_loss': []
        }

        new_net = deepcopy(self.net)
        y_hat = new_net(spt_x)

        loss = F.mse_loss(y_hat, spt_y)
        loss_set['train_loss'].append(loss.item())
        if loss.item() < min_train_loss:
            min_train_loss = loss.item()
        grad = torch.autograd.grad(loss, new_net.parameters())
        grads_params = zip(grad, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.fine_tune_lr * p[0], grads_params))

        with torch.no_grad():
            y_hat = new_net(qry_x, fast_weights)
            loss_qry = F.mse_loss(y_hat, qry_y)
            loss_set['validation_loss'].append(loss_qry.item())
            loss_qry_list.append(loss_qry)


            rmse, smape = Metrics(qry_y, y_hat)

            rmse_list.append(rmse)
            smape_list.append(smape)
            min_rmse = rmse
            min_smape = smape
            min_loss = loss_qry.item()
            rmse_best_epoch = 1
            smape_best_epcoh = 1

        if naive:
            print('    Epoch [1] | train_loss: %.4f | test_loss: %.4f | rmse: %.4f | smape: %.4f |'
                  % (loss.item(), loss_qry.item(), rmse, smape))

        for step in range(1, self.update_step_test):
            y_hat = new_net(spt_x, fast_weights)
            loss = F.mse_loss(y_hat, spt_y)
            loss_set['train_loss'].append(loss.item())
            if loss.item() < min_train_loss:
                min_train_loss = loss.item()
            grad = torch.autograd.grad(loss, fast_weights)
            grads_params = zip(grad, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.fine_tune_lr * p[0], grads_params))

            # testing on query set
            with torch.no_grad():
                # calculating metrics
                y_hat = new_net(qry_x, fast_weights)
                loss_qry = F.mse_loss(y_hat, qry_y)
                loss_set['validation_loss'].append(loss_qry.item())
                loss_qry_list.append(loss_qry)

                rmse, smape = Metrics(qry_y, y_hat)

                rmse_list.append(rmse)
                smape_list.append(smape)

                if min_smape > smape:
                    min_smape = smape
                    smape_best_epcoh = step + 1
                if min_rmse > rmse:
                    min_loss = loss_qry.item()
                    min_rmse = rmse
                    rmse_best_epoch = step + 1
                    print(
                                '    Epoch [%d] | train_loss: %.4f | test_loss: %.4f | rmse: %.4f | smape: %.4f |'
                                % (step + 1, loss.item(), loss_qry.item(), rmse, smape))

        del new_net

        return {
            'test_loss': min_loss,
            'train_loss': min_train_loss,
            'rmse': min_rmse,
            'smape': min_smape,
            'rmse_best_epoch': rmse_best_epoch,
            'smape_best_epoch': smape_best_epcoh,
            'loss_set': loss_set
        }

    pass


if __name__ == '__main__':
    pass

