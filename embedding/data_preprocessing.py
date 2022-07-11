# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2022-07-11'

'''
the scripts for time series data processing.
'''

import os
import os.path as osp
import torch
import numpy as np
from collections import OrderedDict
from configs import DATA_DIR as DATADIR
from configs import few_shot_dataset_name
from tools.tools import obj_serialization, read_tsv, obj_unserialization, generate_filename
from sklearn import preprocessing


def few_shot_data(path=None):

    if path is None:
        raise Exception('The parameter "path" is None!')
    dataset_file_names = os.listdir(path)
    train_dataset = OrderedDict()
    test_dataset = OrderedDict()
    process_traindata_num = 0
    process_testdata_num = 0
    for dir in dataset_file_names:
        dataset_dir = osp.join(path, dir)
        if os.path.isdir(dataset_dir):
            train_file_path = osp.join(dataset_dir, '%s_TRAIN.tsv' % dir)
            test_file_path = osp.join(dataset_dir, '%s_TEST.tsv' % dir)
            if os.path.isfile(train_file_path):
                train_dataset.setdefault(dir, read_tsv(train_file_path).loc[:, 1:].values.astype(np.float64))
                process_traindata_num += 1
            else:
                print('"%s" is not a file!' % train_file_path)
            if os.path.isfile(test_file_path):
                test_dataset.setdefault(dir, read_tsv(test_file_path).loc[:, 1:].values.astype(np.float64))
                process_testdata_num += 1
            else:
                print('"%s" is not a file!' % test_file_path)

    obj_serialization(osp.join(DATADIR, 'train_data.pkl'), train_dataset)
    obj_serialization(osp.join(DATADIR, 'test_data.pkl'), test_dataset)
    print('train_process_num: %d' % process_traindata_num)
    print('test_process_num: %d' % process_testdata_num)


def split_data(data_path=None, ratio=0.1, shuffle=False, data=None):

    if data is None:
        data = obj_unserialization(data_path)
    if int(ratio) >= len(data):
        return [], []
    if 0 < ratio < 1:
        train_data_size = int(len(data) * ratio)
    elif 1 <= ratio < len(data):
        train_data_size = int(ratio)
    else:
        raise Exception('Invalid value about "ratio" --> [%s]' % str(ratio))

    if train_data_size == 0:
        val_data = data
        train_data = []
    else:
        train_data = data[:train_data_size]
        val_data = data[train_data_size:]
    return train_data, val_data


def create_sequence(data, ratio=0.1):

    forecast_point_num = int(len(data[0]) * ratio)
    position = len(data[0]) - forecast_point_num
    xs = np.array(data)[:, :position]
    ys = np.array(data)[:, position:]

    return torch.from_numpy(xs).float().unsqueeze(dim=2), torch.from_numpy(ys).float(), position, forecast_point_num


def construct_dataset(size=100):

    file_list = os.listdir(DATADIR)

    counter = 0
    data_dict = {}
    file_num = len(file_list)
    for file in file_list:
        data_path = osp.join(DATADIR, file)
        if os.path.isdir(osp.join(DATADIR, file)):
            file_num -= 1
            continue
        data_dict.setdefault(file.split('.')[0], obj_unserialization(data_path))
        counter += 1
        if counter % size == 0:
            save_path = osp.join(DATADIR,
                                     'dataset\\%s' % generate_filename('.pkl', *['UCR', str(counter - size + 1), str(counter)])
                                     )
            obj_serialization(save_path, data_dict)
            print(save_path)
            data_dict.clear()
        elif counter == file_num:
            save_path = osp.join(DATADIR,
                                     'dataset\\%s' % generate_filename('.pkl', *['UCR', str(size * (counter // size) + 1), str(counter)])
                                     )
            obj_serialization(save_path, data_dict)
            print(save_path)
            data_dict.clear()


def normalizer(data=None):

    # z-score normalization

    if data is not None:
        return preprocessing.scale(data, axis=1)
    else:
        raise Exception('data is None!')
    pass


def get_basic_data():

    load_data_path = osp.join(DATADIR, 'few_shot_data\\few_shot_load_data.pkl')
    UCR_train_data_path = osp.join(DATADIR, 'train_data.pkl')
    UCR_test_data_path = osp.join(DATADIR, 'test_data.pkl')

    load_data = obj_unserialization(load_data_path)
    UCR_train_data = obj_unserialization(UCR_train_data_path)
    UCR_test_data = obj_unserialization(UCR_test_data_path)

    few_shot_train_data = OrderedDict()
    few_shot_test_data = OrderedDict()
    for key, value in load_data.items():
        # if key in DIRTY_DATA_ID:
        #     continue
        train_data, test_data = split_data(data=value, ratio=0.5)
        few_shot_train_data.setdefault(key, np.array(train_data))
        few_shot_test_data.setdefault(key, np.array(test_data))
    print('valid load dataset: %d' % len(few_shot_train_data))

    for key, value in UCR_train_data.items():
        if key in few_shot_dataset_name:
            few_shot_train_data.setdefault(key, value)
            few_shot_test_data.setdefault(key, UCR_test_data[key])

    print('all the few shot dataset: %d' % len(few_shot_train_data))

    obj_serialization(osp.join(DATADIR, 'few_shot_data\\train_data.pkl'), few_shot_train_data)
    obj_serialization(osp.join(DATADIR, 'few_shot_data\\test_data.pkl'), few_shot_test_data)


if __name__ == '__main__':

    pass