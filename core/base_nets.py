# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2022/03/10'

'base networks'

# built-in library
import os
import math
import sys
from copy import deepcopy

# third-party library
import torch
import torch.nn as nn
from torch.nn import functional as F

# self-defined library
from tools.tools import metrics as Metrics

torch.set_default_tensor_type(torch.DoubleTensor)


class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.name = 'MLP'

        self.hidden_size = n_hidden
        # this list contains all tensor needed to be optimized
        self.params = nn.ParameterList()

        # linear input layer
        weight = nn.Parameter(torch.ones(n_hidden, n_input))
        bias = nn.Parameter(torch.zeros(n_hidden))
 
        self.params.extend([weight, bias])

        # linear output layer
        weight = nn.Parameter(torch.ones(n_output, n_hidden))
        bias = nn.Parameter(torch.zeros(n_output))

        self.params.extend([weight, bias])

        self.init()

    def parameters(self):
        return self.params

    def init(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, vars=None):

        if vars is None:
            params = self.params
        else:
            params = vars

        # input layer
        (weight_input, bias_input) = (params[0].to(x.device), params[1].to(x.device))
        x = F.linear(x, weight_input, bias_input)

        # output layer
        (weight_output, bias_output) = (params[2].to(x.device), params[3].to(x.device))
        out = F.linear(x, weight_output, bias_output)

        return out


class BaseCNN(nn.Module):

    def __init__(self, output=10):
        super(BaseCNN, self).__init__()
        self.name = 'BASECNN'
        self.output = output

        # this list contains all tensor needed to be optimized
        self.vars = nn.ParameterList()

        # running_mean and running var
        self.vars_bn = nn.ParameterList()

        # 填充需要训练的网络的参数

        # Conv1d layer
        # [channel_out, channel_in, kernel-size]
        weight = nn.Parameter(torch.ones(64, 1, 3))

        nn.init.kaiming_normal_(weight)

        bias = nn.Parameter(torch.zeros(64))

        self.vars.extend([weight, bias])

        # linear layer
        weight = nn.Parameter(torch.ones(self.output, 64*100))
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

        # x = x.squeeze(dim=2)
        # x = x.unsqueeze(dim=1)
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
        # print(n_features)
        self.hidden_size = n_hidden
        self.output_size = n_output
        self.layer_size = n_layer

        # 输入层
        W_i = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.input_size))
        bias_i = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.params.extend([W_i, bias_i])

        # 隐含层
        W_h = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
        bias_h = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.params.extend([W_h, bias_h])

        if self.layer_size > 1:
            for _ in range(self.layer_size - 1):

                # 第i层lstm
                # 输入层
                W_i = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
                bias_i = nn.Parameter(torch.Tensor(self.hidden_size * 4))
                self.params.extend([W_i, bias_i])
                # 隐含层
                W_h = nn.Parameter(torch.Tensor(self.hidden_size * 4, self.hidden_size))
                bias_h = nn.Parameter(torch.Tensor(self.hidden_size * 4))
                self.params.extend([W_h, bias_h])


        # 输出层
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
        # with torch.autograd.set_detect_anomaly(True):
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

            # gates = x_t @ W_i + h_t @ W_h + bias_h + bias_i
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
        # out = hidden_seq[-1] @ W_linear + bias_linear
        return out


class BaseCNNConLSTM(nn.Module):

    def __init__(self,  n_features, n_hidden, n_output, n_layer=1, time_size=1, cnn_feature=200):
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
        x = x.unsqueeze(dim=1)
        output = self.lstm(x, params[self.cnn_tensor_num:], init_states)
        return output
        pass
    pass
