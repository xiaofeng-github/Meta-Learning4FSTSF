# -*- coding:utf-8 -*-
__author__ = 'XF'

'task split'
import sys
import os
import os.path as osp
from torch import from_numpy
import numpy as np

# self-defined tools
from tools import obj_unserialization
from options import task_id_int2str


class LoadData:

    def __init__(self, maml, test_user_index, add_dim_pos, data_path=None, time_size=3, rate=0.8, data_source='Load',
                 ppn=None):

        '''
        :param maml: <bool> whether using first order maml algorithm to train
        :param test_user_index: choose a target task
        :param add_dim_pos:
        :param data_path:
        :param time_size:
        :param rate:
        '''

        if data_path:
            self.indexes = {'train': int(0), 'test': int(0)}
            self.task_id = {'train': [], 'test': []}
            self.features = None
            self.task_num = None
            self.datasets_cache = None
            self.output = 94
            self.outputs = []
            if maml:
                if data_source == 'UCR':
                    self.datasets_cache = self.UCR_data_cache_maml(data_path, test_user_index, time_size, rate,
                                                                   add_dim_pos, data_source, ppn)
                else:
                    self.datasets_cache = self.load_data_cache(data_path, test_user_index, time_size, rate, add_dim_pos)
            elif data_source == 'UCR':
                self.UCR_data_cache(data_path, time_size, add_dim_pos, data_source, ppn)
            else:
                if os.path.isdir(data_path):
                    self.process_batch_file_dataset(data_path, time_size, rate, add_dim_pos, data_source)
                else:
                    self.datasets_cache = self.naive_load_data_cache(data_path, time_size, rate, add_dim_pos,
                                                                     data_source)
        else:
            raise Exception('data dir is None!')
        pass

    def UCR_data_cache(self, data_path, time_size, add_dim_pos, data_source, ppn):

        train_data_path = osp.join(data_path, 'train_data_embedding_%s.pkl' % str(ppn))
        test_data_path = osp.join(data_path, 'test_data_embedding_%s.pkl' % str(ppn))
        self.datasets_cache = {'train': [], 'test': []}

        train_data = obj_unserialization(train_data_path)
        test_data = obj_unserialization(test_data_path)
        for key in train_data.keys():
            train_x, train_y, _ = self.split_x_y(train_data[key], time_size, add_dim_pos, data_source=data_source,
                                                 ratio4UCR=ppn)
            test_x, test_y, _ = self.split_x_y(test_data[key], test_data, add_dim_pos, data_source=data_source,
                                               ratio4UCR=ppn)
            self.datasets_cache['train'].append([train_x, train_y])
            self.datasets_cache['test'].append([test_x, test_y])
            self.task_id['train'].append(key)
            self.task_id['test'].append(key)

        self.task_num = len(self.datasets_cache['train'])

        pass

    def process_batch_file_dataset(self, data_path, time_size, rate, add_dim_pos, data_source):

        # data_path is a dir!
        file_list = os.listdir(data_path)
        self.datasets_cache = {'train': [], 'test': []}
        for file in file_list:
            temp = self.naive_load_data_cache(osp.join(data_path, file), time_size, rate, add_dim_pos, data_source)
            self.datasets_cache['train'].extend(temp['train'])
            self.datasets_cache['test'].extend(temp['test'])
        print(len(self.datasets_cache['train']))
        print(len(self.datasets_cache['test']))
        self.task_num = len(self.datasets_cache['train'])

        pass

    def naive_load_data_cache(self, data_path, time_size, rate, add_dim_pos, data_source):

        datasets_cache = {'train': [], 'test': []}

        data = obj_unserialization(data_path)
        train_data = []
        for key, value in data.items():
            train_data.append(value)
            # print(key)
            self.task_id['train'].append(key.split('_')[0])
            self.task_id['test'].append(key.split('_')[0])

            # ============================================== #
            # update time: 2021-05-01
            # for each dataset only using first category
            break
            # ============================================== #

        self.task_num = len(train_data)
        train_set, test_set = self.split_spt_qry(train_data, rate)

        for train_task, test_task in zip(train_set, test_set):
            train_task_x, train_task_y, _ = self.split_x_y(train_task, time_size, add_dim_pos, data_source)
            test_task_x, test_task_y, _ = self.split_x_y(test_task, time_size, add_dim_pos, data_source)
            datasets_cache['train'].append([train_task_x, train_task_y])
            datasets_cache['test'].append([test_task_x, test_task_y])

        return datasets_cache

    def UCR_data_cache_maml(self, data_path, test_task_index, time_size, rate, add_dim_pos, data_source, ppn):

        datasets_cache = {'train': [], 'test': []}
        train_data_path = osp.join(data_path, 'train_data_embedding_%s.pkl' % str(ppn))
        test_data_path = osp.join(data_path, 'test_data_embedding_%s.pkl' % str(ppn))

        train_data = obj_unserialization(train_data_path)
        test_data = obj_unserialization(test_data_path)

        assert 0 < test_task_index <= len(train_data)
        for i, key in enumerate(train_data.keys()):
            if i == test_task_index - 1:
                test_spt_x, test_spt_y, _ = self.split_x_y(train_data[key], time_size, add_dim_pos,
                                                           data_source=data_source, ratio4UCR=ppn)
                test_qry_x, test_qry_y, _ = self.split_x_y(test_data[key], time_size, add_dim_pos,
                                                           data_source=data_source, ratio4UCR=ppn)
                datasets_cache['test'].append([test_spt_x, test_spt_y, test_qry_x, test_qry_y])
                self.task_id['test'].append(key)
                continue

            train_spt_x, train_spt_y, _ = self.split_x_y(train_data[key], time_size, add_dim_pos,
                                                         data_source=data_source, ratio4UCR=ppn)
            train_qry_x, train_qry_y, _ = self.split_x_y(test_data[key], time_size, add_dim_pos,
                                                         data_source=data_source, ratio4UCR=ppn)
            datasets_cache['train'].append([train_spt_x, train_spt_y, train_qry_x, train_qry_y])
            self.task_id['train'].append(key)

        self.task_num = len(datasets_cache['train'])

        return datasets_cache

    def load_data_cache(self, data_path, test_user_index, time_size, rate, add_dim_pos):

        datasets_cache = {'train': [], 'test': []}

        data = obj_unserialization(data_path)
        assert 0 < test_user_index <= len(data)

        key = task_id_int2str(test_user_index)
        test_data = data[key]
        self.task_id['test'].append(key)
        data.pop(key)
        train_data = []
        for key, value in data.items():
            train_data.append(value)
            self.task_id['train'].append(key)
        self.task_num = len(train_data)
        train_spt, train_qry = self.split_spt_qry(train_data, rate)
        test_spt, test_qry = self.split_spt_qry([test_data], rate)

        for spt, qry in zip(train_spt, train_qry):
            train_spt_x, train_spt_y, _ = self.split_x_y(spt, time_size, add_dim_pos)
            train_qry_x, train_qry_y, _ = self.split_x_y(qry, time_size, add_dim_pos)
            datasets_cache['train'].append([train_spt_x, train_spt_y, train_qry_x, train_qry_y])

        for spt, qry in zip(test_spt, test_qry):
            test_spt_x, test_spt_y, _ = self.split_x_y(spt, time_size, add_dim_pos)
            test_qry_x, test_qry_y, _ = self.split_x_y(qry, time_size, add_dim_pos)
            datasets_cache['test'].append([test_spt_x, test_spt_y, test_qry_x, test_qry_y])

        return datasets_cache

    def get_data(self, task_id=None):

        if task_id in self.task_id['train']:

            # task_id = task_id_int2str(task_id)
            pos_train = self.task_id['train'].index(task_id)
            pos_test = self.task_id['test'].index(task_id)
            return self.datasets_cache['train'][pos_train], self.datasets_cache['test'][pos_test], task_id
        else:
            raise Exception('Unknown the task id [%s]!' % task_id)
        pass

    def split_spt_qry(self, data, rate):

        spt = []
        qry = []
        for task in data:
            pos = int(len(task) * rate)
            spt.append(task[:pos])
            qry.append(task[pos:])

        return spt, qry

    def split_x_y(self, data, time_size, add_dim_pos, data_source='Load', ratio4UCR=10):

        if data_source == 'Load':
            xs = []
            ys = []
            features = len(data[0])
            self.features = features
            for i in range(len(data) - time_size - 1):
                x = data[i: i + time_size]
                y = data[i + time_size]
                xs.append(x)
                ys.append(y)

            if add_dim_pos == -1:
                np_xs = np.array(xs)
            else:
                np_xs = from_numpy(np.array(xs)).unsqueeze(dim=add_dim_pos).numpy()
        else:
            # forecast_point_num = int(len(data[0]) * ratio4UCR)
            forecast_point_num = ratio4UCR
            position = len(data[0]) - forecast_point_num
            xs = np.array(data)[:, :position]
            ys = np.array(data)[:, position:]
            np_xs = from_numpy(xs).unsqueeze(dim=2).numpy()
            self.features = 1
            self.output = forecast_point_num
            self.outputs.append(forecast_point_num)

        return np_xs, np.array(ys), self.features

    def next(self, mode='train'):
        if self.indexes[mode] == len(self.datasets_cache[mode]):
            self.indexes[mode] = 0

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        task_id = self.task_id[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch, task_id


if __name__ == '__main__':

    pass