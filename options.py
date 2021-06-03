# -*- coding:utf-8 -*-
__author__ = 'XF'

'options'
import os
import os.path as osp
import argparse
from builtins import print as b_print
from configs import model_save_dir, loss_save_dir, log_save_path, MODEL_NAME, MODE_NAME, exp_result_dir
from tools import generate_filename


def task_id_int2str(int_id):

    if int_id < 10:
        str_id = '000' + str(int_id)
    elif int_id < 100:
        str_id = '00' + str(int_id)
    elif int_id < 1000:
        str_id = '0' + str(int_id)
    else:
        str_id = str(int_id)

    return str_id


def print(*args, file='./log.txt', end='\n', terminate=True):

    with open(file=file, mode='a', encoding='utf-8') as console:
        b_print(*args, file=console, end=end)
    if terminate:
        b_print(*args, end=end)


def parse_args(script='main'):

    parser = argparse.ArgumentParser(description='Load Forecasting script %s.py' % script)

    # training arguments
    parser.add_argument('--model', default=None,
                        help='the model name is used to train.[lstm, cnn, cnn->lstm, lstm+maml, cnn+maml, cnnConlstm+maml]')
    parser.add_argument('--epoch', type=int, default=10, help='the iteration number for training data.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')

    # data arguments
    parser.add_argument('--dataSet', default='few_shot_data', help='the data path for training and testing.')
    parser.add_argument('--ratio', type=float, default=0.8, help='the ratio of training set for all data set')
    parser.add_argument('--trainSet', default='', help='the path of the training data.')
    parser.add_argument('--testSet', default='', help='the path of the testing data.')
    parser.add_argument('--UCR', action='store_true', default=False, help='for UCR data.')
    parser.add_argument('--ppn', type=int, default=10, help='predict point num.')

    # save-path arguments
    parser.add_argument('--msd', default=model_save_dir, help='the model save dir.')
    parser.add_argument('--lsd', default=loss_save_dir, help='the loss save dir.')
    parser.add_argument('--log', default=log_save_path, help='the log save path.')
    parser.add_argument('--maml_log', default=log_save_path, help='the log save path.')
    parser.add_argument('--rmse_path', default=log_save_path, help='the log save metric rmse.')
    parser.add_argument('--smape_path', default=log_save_path, help='the log save metric smape.')
    # the arguments for LSTM model
    parser.add_argument('--time_size', type=int, default=1, help='the time_size for lstm input')

    # the arguments for cnn model
    parser.add_argument('--add_dim_pos', type=int, default=1, help='the position for add dimension when change sequence data to img data')

    # the implement mode
    parser.add_argument('--mode', default='together',
                        help='the implement mode for script, [training, testing, together] can be chosen')

    # for testing mode
    parser.add_argument('--model_state', default='', help='the path of trained model')

    # for maml
    parser.add_argument('--user_id', type=str, default='0', help='the id of true target task')
    parser.add_argument('--update_step', type=int, default=5, help='the train task update step')
    parser.add_argument('--update_step_test', type=int, default=50, help='the target task update step')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='the learning rate of meta network')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='the learning rate of base network')
    parser.add_argument('--fine_lr', type=float, default=0.03, help='the learning rate of fine tune target network')
    parser.add_argument('--baseNet', default='cnn', help='the base network for maml training, [lstm, cnn, cnnConlstm] can be chosen')
    parser.add_argument('--maml', action='store_true', default=False, help='whether using maml algorithm to train a network.')
    parser.add_argument('--all_data_dir', default=None, help='the directory for all load data.')
    parser.add_argument('--begin_task', type=int, default=1, help='the begining task id that be used to batch training when having maml')
    parser.add_argument('--end_task', type=int, default=12, help='the ending task id that be used to batch training when having maml.')
    parser.add_argument('--batch_task_num', type=int, default=5, help='batch training for maml')
    # for hardware setting
    parser.add_argument('--device', default='cuda', help='the calculate device for torch Tensor, [cpu, cuda] can be chosen')

    params = parser.parse_args()

    # maml log
    if params.maml:
        maml_log_path = osp.join('./maml_log', generate_filename('.txt', *['log'], timestamp=True))
        params.maml_log = maml_log_path
        params.log = maml_log_path
        params_show(params)

    # dynamic generate log file
    log_path = osp.join('./log', generate_filename('.txt', *['log'], timestamp=True))
    params.log = log_path
    params_show(params)

    # generate experimental result log file
    result_dir_name = params.model + '_' + str(params.ppn)
    result_dir = osp.join(exp_result_dir, result_dir_name)
    if not osp.exists(result_dir):
        os.mkdir(result_dir)
    params.rmse_path = osp.join(result_dir, generate_filename('.txt', *['rmse'], timestamp=True))
    params.smape_path = osp.join(result_dir, generate_filename('.txt', *['smape'], timestamp=True))

    # parameters check

    if params.mode == 'together':
        assert float(0) < params.ratio < float(1)  # check for split ratio
    elif params.mode == 'training':
        assert osp.exists(params.trainSet)
    elif params.mode == 'testing':
        assert osp.exists(params.testSet) and osp.exists(params.model_state)
    else:
        raise Exception('Unknown implement mode: %s' % params.mode)

    if params.model in MODEL_NAME:
        assert params.epoch > 0 and isinstance(params.epoch, int)  # check for epoch
        if params.model[:4] == 'lstm' and params.model[-4:] == 'lstm':      # if model is 'lstm', the time_size id needed parameters
            assert params.time_size > 0 and isinstance(params.time_size, int)
    else:
        raise Exception('Unknown model name: %s' % params.model)

    return params


def params_show(params):

    if params:
        print('Parameters Show', file=params.log)
        print('=======================================', file=params.log)
        print('About model:', file=params.log)
        print('    model: %s' % params.model, file=params.log)
        print('    epoch: %s' % params.epoch, file=params.log)
        print('    learning rate: %s' % str(params.lr), file=params.log)
        if params.model == 'lstm':
            print('    time size: %d' % params.time_size, file=params.log)
        if params.mode == 'testing':
            print('    trained model path: %s' % params.model_state, file=params.log)
        print('About data:', file=params.log)
        print('    data file: %s' % params.dataSet, file=params.log)
        print('    training data file: %s' % params.trainSet, file=params.log)
        print('    testing data file: %s' % params.testSet, file=params.log)
        print('    data split rate: %s' % params.ratio, file=params.log)
        print('    predict point num: %d' % params.ppn, file=params.log)
        print('implement mode: %s' % params.mode, file=params.log)

        print('=======================================', file=params.log)
        print('MAML Show', file=params.log)
        print('=======================================', file=params.log)
        if params.user_id != '0':
            #target_task = task_id_int2str(params.user_id)
            target_task = params.user_id
            print('    target task: %s' % target_task, file=params.log)
        else:
            begin_task = task_id_int2str(params.begin_task)
            end_task = task_id_int2str(params.end_task)
            print('    begin task: %s' % begin_task, file=params.log)
            print('    end task: %s' % end_task, file=params.log)
        print('    update step: %d' % params.update_step, file=params.log)
        print('    update step test: %d' % params.update_step_test, file=params.log)
        print('    meta lr: %.4f' % params.meta_lr, file=params.log)
        print('    base lr: %.4f' % params.base_lr, file=params.log)
        print('    fine lr: %.4f' % params.fine_lr, file=params.log)
        print('    device: %s' % params.device, file=params.log)
        if params.maml:
            print('    MAML: True', file=params.log)
        else:
            print('    MAML: False', file=params.log)
        print('=======================================', file=params.log)
    else:
        raise Exception('params is None!', file=params.log)
    pass


if __name__ == '__main__':
    pass





