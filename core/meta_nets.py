# -*- coding:utf-8 -*-
__author__ = 'XF'
__date___ = '2022/03/10'

'meta networks'

# built-in library
import os.path as osp
from copy import deepcopy

# third-party library
import torch
import torch.nn as nn
from torch.nn import functional as F

# self-defined library
from tools.tools import metrics as Metrics, generate_filename
from configs import MODEL_PATH

torch.set_default_tensor_type(torch.DoubleTensor)


class MetaNet(nn.Module):

    def __init__(self, baseNet=None, update_step_train=10, update_step_target=20, meta_lr=0.001, base_lr=0.01, fine_lr=0.01):

        super(MetaNet, self).__init__()
        self.update_step_train = update_step_train
        self.update_step_target = update_step_target
        self.meta_lr = meta_lr
        self.base_lr = base_lr
        self.fine_tune_lr = fine_lr

        if baseNet is not None:
            self.net = baseNet
        else:
            raise Exception('baseNet is None')
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.meta_optim = torch.optim.SGD(self.net.parameters(), lr=self.meta_lr)
        pass
    
    def save_model(self, model_name='model'):
        torch.save(self.net, osp.join(MODEL_PATH, generate_filename('pth',*[model_name,])))

    def forward(self, spt_x, spt_y, qry_x, qry_y, device='cpu'):
        '''

        :param spt_x: if baseNet is cnn: [ spt size, in_channel, height, width], lstm [spt_size, time_size, feature_size]
        :param spt_y: [ spt size]
        :param qry_x: if baseNet is cnn: [ qry size, in_channel, height, width], lstm [qry size, time_size, feature_size]
        :param qry_y: [ qry size]
        :param min_max_data_path: 用来进行数据反归一化的min,max值的存储路径
        :return:
        batch size 在本任务中设置为1， 即每次采样一个任务进行训练
        '''

        # spt_size, channel, height, width = spt_x.size()
        # qry_size = spt_y.size(0)
        task_num = len(spt_x)
        loss_list_qry = []
        mape_list = []
        rmse_list = []
        smape_list = []
        qry_loss_sum = 0
        # print('更新任务网络===============================================')
        # 第0步更新
        for i in range(task_num):
            x_spt = torch.from_numpy(spt_x[i]).to(device)
            y_spt = torch.from_numpy(spt_y[i]).to(device)
            x_qry = torch.from_numpy(qry_x[i]).to(device)
            y_qry = torch.from_numpy(qry_y[i]).to(device)

            y_hat = self.net(x_spt, vars=None)
            loss = F.mse_loss(y_hat, y_spt)
            grad = torch.autograd.grad(loss, self.net.parameters())
            grads_params = zip(grad, self.net.parameters())  # 将梯度和参数一一对应起来

            # fast_weights 这一步相当于求了一个 theta - alpha * nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], grads_params))

            # 在query集上测试，计算准确率
            # 使用更新后的参数在query集上测试
            with torch.no_grad():
                y_hat = self.net(x_qry, fast_weights)
                loss_qry = F.mse_loss(y_hat, y_qry)
                loss_list_qry.append(loss_qry)

                # 计算评价指标
                rmse, mape, smape = Metrics(y_qry, y_hat)

                rmse_list.append(rmse)
                mape_list.append(mape)
                smape_list.append(smape)

            for step in range(1, self.update_step_train):
                y_hat = self.net(x_spt, fast_weights)
                loss = F.mse_loss(y_hat, y_spt)
                grad = torch.autograd.grad(loss, fast_weights)
                grads_params = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], grads_params))

                if step < self.update_step -1:
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
                    rmse, mape, smape = Metrics(y_qry, y_hat)

                    rmse_list.append(rmse)
                    mape_list.append(mape)
                    smape_list.append(smape)
                    pass

        # 更新元网络
        loss_qry = qry_loss_sum / task_num  # 表示在经过update_step之后，learner在当前任务query set上的损失
        self.meta_optim.zero_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()

        
        return {
            'loss': loss_list_qry[-1].item(), 
            'rmse': rmse_list[-1], 
            'mape': mape_list[-1], 
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

        # 评价指标
        loss_qry_list = []
        rmse_list = []
        mape_list = []
        smape_list = []
        min_loss = 0
        best_epoch = 0
        min_train_loss = 1000000
        loss_set = {
            'train_loss': [],
            'validation_loss': []
        }

        # new_net = deepcopy(self.net)
        # new_net = self.net
        y_hat = self.net(spt_x)
        # with torch.autograd.set_detect_anomaly(True):
        loss = F.mse_loss(y_hat, spt_y)
        loss_set['train_loss'].append(loss.item())
        if loss.item() < min_train_loss:
            min_train_loss = loss.item()
        grad = torch.autograd.grad(loss, self.net.parameters())
        grads_params = zip(grad, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.fine_tune_lr * p[0], grads_params))

        # 在query集上测试，计算评价指标
        # 使用更新后的参数进行测试
        with torch.no_grad():
            y_hat = self.net(qry_x, fast_weights)
            loss_qry = F.mse_loss(y_hat, qry_y)
            loss_set['validation_loss'].append(loss_qry.item())
            loss_qry_list.append(loss_qry)
            # 计算评价指标mape
            rmse, mape, smape = Metrics(qry_y, y_hat)

            rmse_list.append(rmse)
            mape_list.append(mape)
            smape_list.append(smape)
            min_rmse = rmse
            min_mape = mape
            min_smape = smape
            min_loss = loss_qry.item()
            rmse_best_epoch = 1
            mape_best_epoch = 1
            smape_best_epcoh = 1
 
        if naive:
            print('    Epoch [1] | train_loss: %.4f | test_loss: %.4f | rmse: %.4f | mape: %.4f | smape: %.4f |'
                  % (loss.item(), loss_qry.item(), rmse, mape, smape))

        for step in range(1, self.update_step_target):
            y_hat = self.net(spt_x, fast_weights)
            loss = F.mse_loss(y_hat, spt_y)
            loss_set['train_loss'].append(loss.item())
            if loss.item() < min_train_loss:
                min_train_loss = loss.item()
            grad = torch.autograd.grad(loss, fast_weights)
            grads_params = zip(grad, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.fine_tune_lr * p[0], grads_params))

            # 在query测试
            with torch.no_grad():
                # 计算评价指标
                y_hat = self.net(qry_x, fast_weights)
                loss_qry = F.mse_loss(y_hat, qry_y)
                loss_set['validation_loss'].append(loss_qry.item())
                loss_qry_list.append(loss_qry)

                rmse, mape, smape = Metrics(qry_y, y_hat)

                rmse_list.append(rmse)
                mape_list.append(mape)
                smape_list.append(smape)
                if min_rmse > rmse:
                    min_rmse = rmse
                    rmse_best_epoch = step + 1
                if min_smape > smape:
                    min_smape = smape
                    smape_best_epcoh = step + 1
                    min_rmse = rmse
                    self.save_model(model_name=self.net.name)
                    print('    Epoch [%d] | train_loss: %.4f | test_loss: %.4f | rmse: %.4f | smape: %.4f |'
                          % (step + 1, loss.item(), loss_qry.item(), rmse, smape))
                    
        return {
            'test_loss': min_loss,
            'train_loss': min_train_loss,
             'rmse': min_rmse,
            'mape': min_mape,
            'smape': min_smape,
            'rmse_best_epoch': rmse_best_epoch,
            'mape_best_epoch': mape_best_epoch,
            'smape_best_epoch': smape_best_epcoh,
            'loss_set': loss_set
                }
    pass