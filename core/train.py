# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2022/03/10'

'train base network + maml'

# built-in library
import os
import os.path as osp
import time
import copy
from datetime import datetime

# third-party library
import torch
import numpy as np

# self-defined library
from core.base_nets import MLP, BaseLSTM, BaseCNN, BaseCNNConLSTM
from core.meta_nets import MetaNet
from core.task_split import LoadData
from configs import TRAINING_TASK_SET
from tools.tools import generate_filename, obj_serialization
from core.options import print as write_log

torch.set_default_tensor_type(torch.DoubleTensor)
    


def train(epoch_num, test_user_index, add_dim_pos, data_path, update_step_train, 
            update_step_target, meta_lr, base_lr, fine_lr, device, baseNet, 
            maml, log, maml_log, lsd, ppn, rmse_path, mape_path, 
            smape_path, ft_step, batch_task_num=5, new_settings=False):

    # 设置随机数种子，保证运行结果可复现
    torch.manual_seed(1)
    np.random.seed(1)
    if device == 'cuda':
        torch.cuda.manual_seed_all(1)

    device = torch.device(device)  # 选择 torch.Tensor 进行运算时的设备对象['cpu', 'cuda']
    if baseNet == 'mlp':
        #print('no add dimension')
        add_dim_pos = -1  # 不给tensor添加维度
    
    # get data
    data = LoadData(maml, test_user_index, add_dim_pos, data_path=data_path, ppn=ppn)

    if baseNet == 'cnn':
        BaseNet = BaseCNN(output=data.output)
    elif baseNet == 'lstm':
        BaseNet = BaseLSTM(n_features=data.features, n_hidden=100, n_output=data.output)
    elif baseNet == 'cnnConlstm':
        BaseNet = BaseCNNConLSTM(n_features=data.features, n_hidden=100, n_output=data.output, cnn_feature=200)
    # ========================================= #
    # update time: 2021-12-10
    elif baseNet == 'mlp':
        # print(data.features, data.output)
        BaseNet = MLP(n_input=data.features, n_hidden=100, n_output=data.output)
    # ======================================== #
    else:
        raise Exception('Unknown baseNet: %s' % baseNet)


    metaNet = MetaNet(
        baseNet=BaseNet,
        update_step_train=update_step_train,
        update_step_target=update_step_target,
        meta_lr=meta_lr,
        base_lr=base_lr,
        fine_lr=fine_lr
                    ).to(device)

    training_result = {
        'target task': None,
        'qry_loss': None,
        'rmse': None,
        'mape': None,
        'smape': None,
        'rmse_best_epoch': None,
        'mape_best_epoch': None,
        'smape_best_epoch': None,
        'training time:': None,
        'date': None,
        'log': log,
        'maml_log': None
    }

    # training
    start = time.time()
    step = 0
    batch_num = 0
    train_loss = {}
    test_loss = {}
    # print(data.task_num)

    if maml:
        while step < epoch_num:

            (spt_x, spt_y), (qry_x, qry_y), task_id = batch_task(data, batch_task_num=batch_task_num)

            batch_num += 1

            print('[%d]===================== training 元网络========================= :[%s]' % (step, task_id))
            metrics = metaNet(spt_x, spt_y, qry_x, qry_y,device=device)
            print('| train_task: %s | qry_loss: %.4f | qry_rmse: %.4f | qry_mape: %.4f | qry_smape: %.4f |' 
            % (task_id, metrics['loss'], metrics['rmse'], metrics['mape'], metrics['smape']))


            if batch_num % (data.task_num // batch_task_num) == 0:
                step += 1
                (spt_x, spt_y, qry_x, qry_y), task_id = data.next('test')

                spt_x, spt_y, qry_x, qry_y = torch.from_numpy(spt_x).to(device), \
                                             torch.from_numpy(spt_y).to(device), \
                                             torch.from_numpy(qry_x).to(device), \
                                             torch.from_numpy(qry_y).to(device)
                print('===================== fine tuning 目标网络 ========================= :[%s]' % task_id)
                metrics = metaNet.fine_tuning(spt_x, spt_y, qry_x, qry_y)

                if training_result['qry_loss'] is None:
                    training_result['target task'] = task_id
                    training_result['qry_loss'] = metrics['test_loss']
                    training_result['mape'] = metrics['mape']
                    training_result['smape'] = metrics['smape']
                    training_result['rmse'] = metrics['rmse']
                    training_result['rmse_best_epoch'] = metrics['rmse_best_epoch']
                    training_result['mape_best_epoch'] = metrics['mape_best_epoch']
                    training_result['smape_best_epoch'] = metrics['smape_best_epoch']
                else:
                    if training_result['rmse'] > metrics['rmse']:
                        training_result['qry_loss'] = metrics['test_loss']
                        training_result['rmse'] = metrics['rmse']
                        training_result['rmse_best_epoch'] = metrics['rmse_best_epoch']
                    if training_result['mape'] > metrics['mape']:
                        training_result['mape'] = metrics['mape']
                        training_result['mape_best_epoch'] = metrics['mape_best_epoch']
                    if training_result['smape'] > metrics['smape']:
                        training_result['smape'] = metrics['smape']
                        training_result['smape_best_epoch'] = metrics['smape_best_epoch']
                write_log(
                    'Epoch [%d] | '
                    'target_task_id: %s | '
                    'qry_loss: %.4f | '
                    'rmse: %.4f(%d) | '
                    'smape: %.4f(%d) |'
                    % (
                        step,
                        task_id,
                        metrics['test_loss'],
                        metrics['rmse'], metrics['rmse_best_epoch'],
                        metrics['smape'], metrics['smape_best_epoch']
                    ),
                    file=log
                )
    else:

        # updated date: 2021-10-20
        if not new_settings:
            (train_x, train_y), (test_x, test_y), task_id = data.get_data(test_user_index)
        else:
            (train_x, train_y), (test_x, test_y), task_id = new_settings_get_training_data(data,test_user_index)
        # ============================================= #


        spt_x, spt_y, qry_x, qry_y = torch.from_numpy(train_x).to(device), \
                                        torch.from_numpy(train_y).to(device), \
                                        torch.from_numpy(test_x).to(device), \
                                        torch.from_numpy(test_y).to(device)

        print('===================== training %s ========================= :[%s]' % (baseNet, task_id))
        metrics = metaNet.fine_tuning(spt_x, spt_y, qry_x, qry_y, naive=True)
        # ============================================= #
        # updated date: 2021-10-20
        if new_settings:
            print('=====================new settings finetuning=====================')
            (train_x, train_y), (test_x, test_y), task_id = new_settings_get_finetune_data(data,test_user_index)
            spt_x, spt_y, qry_x, qry_y = torch.from_numpy(train_x).to(device), \
                                        torch.from_numpy(train_y).to(device), \
                                        torch.from_numpy(test_x).to(device), \
                                        torch.from_numpy(test_y).to(device)
            metaNet.update_step_target = ft_step
            metrics = metaNet.fine_tuning(spt_x, spt_y, qry_x, qry_y, naive=True) 
        # ============================================= #

        save_loss(lsd, metrics['loss_set'], *[task_id, baseNet, 'loss_set'])
        train_loss.setdefault(task_id, metrics['train_loss'])
        test_loss.setdefault(task_id, metrics['test_loss'])
        write_log(
            'target_task_id: %s | '
            'spt_loss: %.4f |'
            'qry_loss: %.4f | '
            'rmse: %.4f(%d)| '
            'smape: %.4f(%d)|'
            % (
                task_id,
                metrics['train_loss'],
                metrics['test_loss'],
                metrics['rmse'], metrics['rmse_best_epoch'],
                metrics['smape'], metrics['smape_best_epoch']
            ),
            file=log
        )
        # save metrics rmse, mape, smape
        write_log('%.4f (%d)' % (metrics['rmse'], metrics['rmse_best_epoch']), file=rmse_path, terminate=False)
        write_log('%.4f (%d)' % (metrics['mape'], metrics['mape_best_epoch']), file=mape_path, terminate=False)
        write_log('%.4f (%d)' % (metrics['smape'], metrics['smape_best_epoch']), file=smape_path, terminate=False)
        pass
    end = time.time()
    # save loss
    save_loss(lsd, train_loss, *[baseNet, 'train', 'loss'])
    save_loss(lsd, test_loss, *[baseNet, 'test', 'loss'])
    training_result['training time'] = '%s Min' % str((end - start) / 60)
    training_result['date'] = datetime.strftime(datetime.now(), '%Y/%m/%d %H:%M:%S')
    if maml:
        training_result['maml_log'] = maml_log
        train_result(training_result, file=maml_log)
        # save metrics rmse, mape, smape
        write_log('%.4f (%d)' % (training_result['rmse'], training_result['rmse_best_epoch']), file=rmse_path, terminate=False)
        write_log('%.4f (%d)' % (training_result['mape'], training_result['mape_best_epoch']), file=mape_path, terminate=False)
        write_log('%.4f (%d)' % (training_result['smape'], training_result['smape_best_epoch']), file=smape_path, terminate=False)
    else:
        # train_result(training_result, file=log)
        pass
        
    return metrics['smape'], metrics['rmse']


def save_loss(lsd, obj, *others):

    if not osp.exists(lsd):
        os.mkdir(lsd)
    loss_path = osp.join(lsd, generate_filename('.pkl', *others, timestamp=False))
    obj_serialization(loss_path, obj)
    print('loss serialization is finished!')


def train_result(data_dict, file='./log.txt'):
    write_log('training result:============================', file=file)
    for key, value in data_dict.items():
        write_log('    %s: %s' % (key, value), file=file)
    write_log('============================================', file=file)


def batch_task(data, batch_task_num=1, ablation=1):

    # abalation == 1: means that uses all tasks as training task set
    # abalation == 0: menas that only uses UCR tasks as training task set

    (spt_x, spt_y, qry_x, qry_y), task_id = data.next('train')
    train_x = list([spt_x])
    train_y = list([spt_y])
    test_x = list([qry_x])
    test_y = list([qry_y])

    if ablation == 1:
        while batch_task_num > 1:
            (x1, y1, x2, y2), temp  = data.next('train')
            train_x.append(x1)
            train_y.append(y1)
            test_x.append(x2)
            test_y.append(y2)
            task_id += ('-' + temp)
            batch_task_num -= 1
    elif ablation == 0:
        while batch_task_num > 1:
            (x1, y1, x2, y2), temp  = data.next('train')
            if temp.isdigit():
                continue
            train_x.append(x1)
            train_y.append(y1)
            test_x.append(x2)
            test_y.append(y2)
            task_id += ('-' + temp)
            batch_task_num -= 1
    else:
        raise Exception('UnKnown abalaion code: [%d]' % ablation)
    
    return (train_x, train_y), (test_x, test_y), task_id
# ==================================================================================== #
# updated date: 2021-10-20
# in light of reviewer's suggestions, add a group of experiments settings

def new_settings_get_training_data(data, target_task):
    
    # data: UCR
    # 将除target_task以外的其他的数据集打包到一起进行训练
    
    dataset_list = copy.deepcopy(TRAINING_TASK_SET)
    dataset_list.remove(target_task)

    (train_x, train_y), (test_x, test_y), _ = data.get_data(dataset_list[0])
    #print(dataset_list[0])
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    for dataset in dataset_list[1:]:
        # print(dataset)
        (temp_1, temp_2), (temp_3, temp_4), _ = data.get_data(dataset)
        # print(temp_1.shape, temp_2.shape, temp_3.shape, temp_4.shape)
        train_x = np.concatenate((train_x, temp_1), axis=0)
        train_y = np.concatenate((train_y, temp_2), axis=0)
        test_x = np.concatenate((test_x, temp_3), axis=0)
        test_y = np.concatenate((test_y, temp_4), axis=0)
        # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return (train_x, train_y), (test_x, test_y), target_task


def new_settings_get_finetune_data(data, target_task):

    # 用target task 进行微调

    return data.get_data(target_task)

# ================================================================================== #



if __name__ == '__main__':
    
    pass

