# -*- coding:utf-8 -*-
__author__ = 'XF'

'train base network + first order maml'
import os
import os.path as osp
import time
from datetime import datetime
import torch
import numpy as np
from meta_network import MetaNet, BaseLSTM, BaseCNN, BaseCNNConLSTM
from task_split import LoadData
from configs import data_dir, BASE_DIR
from tools import generate_filename, obj_serialization
from options import parse_args, task_id_int2str
from options import print as write_log
torch.set_default_tensor_type(torch.DoubleTensor)


def train(epoch_num, test_user_index, add_dim_pos, data_path, time_size, rate,
          update_step, update_step_test, meta_lr, base_lr, fine_lr, device, baseNet,
          maml, all_data_dir, log, maml_log, lsd, UCR, ppn,
          rmse_path, smape_path, batch_task_num=5):

    # set random seeds, for implement result can be reproduced.
    torch.manual_seed(1)
    np.random.seed(1)
    if device == 'cuda':
        torch.cuda.manual_seed_all(1)

    device = torch.device(device)
    if baseNet == 'lstm':
        add_dim_pos = -1
    data = LoadData(maml, test_user_index, add_dim_pos, data_path=data_path, time_size=time_size, rate=rate,data_source=('UCR' if UCR else 'Load'), ppn=ppn)
    if baseNet == 'cnn':
        BaseNet = BaseCNN(output=data.output)
    elif baseNet == 'lstm':
        BaseNet = BaseLSTM(n_features=data.features, n_hidden=100, n_output=data.output)
        pass
    elif baseNet == 'cnnConlstm':
        BaseNet = BaseCNNConLSTM(n_features=data.features, n_hidden=100, n_output=data.output, cnn_feature=200)
        pass
    else:
        raise Exception('Unknown baseNet: %s' % baseNet)

    metaNet = MetaNet(
        baseNet=BaseNet,
        update_step=update_step,
        update_step_test=update_step_test,
        meta_lr=meta_lr,
        base_lr=base_lr,
        fine_lr=fine_lr
                    ).to(device)

    training_result = {
        'target task': None,
        'qry_loss': None,
        'rmse': None,
        'smape': None,
        'rmse_best_epoch': None,
        'smape_best_epoch': None,
        'training time:': None,
        'date': None,
        'log': log,
        'maml_log': None
    }

    # training
    start = time.time()
    step = 0
    train_loss = {}
    test_loss = {}
    print(data.task_num)
    delete_task_num = 0
    while step < epoch_num:
        if maml:

            (spt_x, spt_y, qry_x, qry_y), task_id = batch_task(data, batch_task_num=batch_task_num)
            step += 1


            print('[%d]===================== training Meta Net========================= :[%s]' % (step, task_id))
            metrics = metaNet(spt_x, spt_y, qry_x, qry_y, device=device)
            print('| train_task: %s | qry_loss: %.4f | qry_rmse: %.4f | qry_smape: %.4f |'
            % (task_id, metrics['loss'], metrics['rmse'], metrics['smape']))

            if step % (data.task_num // batch_task_num) == 0:
                (spt_x, spt_y, qry_x, qry_y), task_id = data.next('test')

                spt_x, spt_y, qry_x, qry_y = torch.from_numpy(spt_x).to(device), \
                                             torch.from_numpy(spt_y).to(device), \
                                             torch.from_numpy(qry_x).to(device), \
                                             torch.from_numpy(qry_y).to(device)
                print('===================== fine tuning Target Net ========================= :[%s]' % task_id)
                metrics = metaNet.fine_tuning(spt_x, spt_y, qry_x, qry_y)

                if training_result['qry_loss'] is None:
                    training_result['target task'] = task_id
                    training_result['qry_loss'] = metrics['test_loss']
                    training_result['smape'] = metrics['smape']
                    training_result['rmse'] = metrics['rmse']
                    training_result['rmse_best_epoch'] = metrics['rmse_best_epoch']
                    training_result['smape_best_epoch'] = metrics['smape_best_epoch']
                else:
                    if training_result['rmse'] > metrics['rmse']:
                        training_result['qry_loss'] = metrics['test_loss']
                        training_result['rmse'] = metrics['rmse']
                        training_result['rmse_best_epoch'] = metrics['rmse_best_epoch']
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
            if step == data.task_num:
                break
            else:
                print('step: %d != task_num:%d' % (step, data.task_num))
            if test_user_index != '0':  # training single task
                (train_x, train_y), (test_x, test_y), task_id = data.get_data(test_user_index)
                step = data.task_num - 1
            else:  # train all tasks
                (train_x, train_y), task_id = data.next('train')
                (test_x, test_y), task_id = data.next('test')

            step += 1

            spt_x, spt_y, qry_x, qry_y = torch.from_numpy(train_x).to(device), \
                                         torch.from_numpy(train_y).to(device), \
                                         torch.from_numpy(test_x).to(device), \
                                         torch.from_numpy(test_y).to(device)
            print('[%d]===================== training %s ========================= :[%s]' % (step, baseNet, task_id))
            metrics = metaNet.fine_tuning(spt_x, spt_y, qry_x, qry_y, naive=True)
            save_loss(lsd, metrics['loss_set'], *[task_id, baseNet, 'loss_set'])
            train_loss.setdefault(task_id, metrics['train_loss'])
            test_loss.setdefault(task_id, metrics['test_loss'])
            write_log(
                'Epoch [%d] | '
                'target_task_id: %s | '
                'spt_loss: %.4f |'
                'qry_loss: %.4f | '
                'rmse: %.4f(%d)| '
                'smape: %.4f(%d)|'
                % (
                    step,
                    task_id,
                    metrics['train_loss'],
                    metrics['test_loss'],
                    metrics['rmse'], metrics['rmse_best_epoch'],
                    metrics['smape'], metrics['smape_best_epoch']
                ),
                file=log
            )
            # save metrics rmse, smape
            write_log('%.4f (%d)' % (metrics['rmse'], metrics['rmse_best_epoch']), file=rmse_path, terminate=False)
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
        # save metrics rmse, smape
        write_log('%.4f (%d)' % (training_result['rmse'], training_result['rmse_best_epoch']), file=rmse_path, terminate=False)
        write_log('%.4f (%d)' % (training_result['smape'], training_result['smape_best_epoch']), file=smape_path, terminate=False)
    else:
        train_result(training_result, file=log)
    pass


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

    return (train_x, train_y, test_x, test_y), task_id




if __name__ == '__main__':

    print('--------------------------------------------Load Forecasting------------------------------------------')
    params = parse_args('train_maml')
    start = time.time()
    if params.user_id == '0' and params.maml:  # training all tasks when maml is true and user_id equal zero
        for user_id in range(params.begin_task, params.end_task + 1):
            train(
                epoch_num=params.epoch,
                test_user_index=user_id,
                add_dim_pos=params.add_dim_pos,
                data_path=osp.join(BASE_DIR, params.dataSet),
                time_size=params.time_size,
                rate=params.ratio,
                update_step=params.update_step,
                update_step_test=params.update_step_test,
                meta_lr=params.meta_lr,
                base_lr=params.base_lr,
                fine_lr=params.fine_lr,
                device=params.device,
                baseNet=params.baseNet,
                maml=params.maml,
                all_data_dir=params.all_data_dir,
                log=params.log,
                maml_log=params.maml_log,
                lsd=params.lsd,
                UCR=params.UCR,
                ppn=params.ppn,
                batch_task_num=params.batch_task_num,
                rmse_path=params.rmse_path,
                smape_path=params.smape_path
            )
        end = time.time()
        print('using time: %.4f Hour' % ((end - start) / 3600.0))
        print('training is over!')
    else:
        train(
            epoch_num=params.epoch,
            test_user_index=params.user_id,
            add_dim_pos=params.add_dim_pos,
            data_path=osp.join(BASE_DIR, params.dataSet),
            time_size=params.time_size,
            rate=params.ratio,
            update_step=params.update_step,
            update_step_test=params.update_step_test,
            meta_lr=params.meta_lr,
            base_lr=params.base_lr,
            fine_lr=params.fine_lr,
            device=params.device,
            baseNet=params.baseNet,
            maml=params.maml,
            all_data_dir=params.all_data_dir,
            log=params.log,
            maml_log=params.maml_log,
            lsd=params.lsd,
            UCR=params.UCR,
            ppn=params.ppn,
            rmse_path=params.rmse_path,
            smape_path=params.smape_path
        )
        end = time.time()
        print('using time: %.4f Min' % ((end - start) / 60.0))
        print('training is over!')
    pass

