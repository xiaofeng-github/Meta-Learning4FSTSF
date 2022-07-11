# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__  = '2022/03/10'

'task split'

# built-in library
import os.path as osp

# third-party library
from torch import from_numpy
import numpy as np

# self-defined tools
from configs import DATA_DIR as data_dir
from tools.tools import obj_unserialization
from configs import TRAINING_TASK_SET


class LoadData:

    def __init__(self, maml, test_user_index, add_dim_pos, data_path=None, ppn=None):

        '''
        :param maml: <bool> 是否使用 maml algorithm to train
        :param test_user_index: 选择一个用户测试，其他用户用来训练meta-network的参数
        :param add_dim_pos: 给tensor增加维度的位置, 当add_dim_pos=-1时，不加维度
        :param data_path: 原始数据路径
        ppn: predict point num
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
                self.datasets_cache = self.UCR_data_cache_maml(data_path, test_user_index, add_dim_pos, ppn)
            else:
                self.UCR_data_cache(data_path, add_dim_pos, ppn)
        else:
            raise Exception('data dir is None!')
        pass

    def UCR_data_cache(self, data_path, add_dim_pos, ppn):

        train_data_path = osp.join(data_path, 'train_data_embedding_%s.pkl' % str(ppn))
        test_data_path = osp.join(data_path, 'test_data_embedding_%s.pkl' % str(ppn))
        self.datasets_cache = {'train': [], 'test': []}

        train_data = obj_unserialization(train_data_path)
        test_data = obj_unserialization(test_data_path)
        for key in train_data.keys():
            train_x, train_y, _ = self.split_x_y(train_data[key], add_dim_pos, ppn=ppn)
            test_x, test_y, _ = self.split_x_y(test_data[key], add_dim_pos, ppn=ppn)
            self.datasets_cache['train'].append([train_x, train_y])
            self.datasets_cache['test'].append([test_x, test_y])
            self.task_id['train'].append(key)
            self.task_id['test'].append(key)

        self.task_num = len(self.datasets_cache['train'])

        pass

    def UCR_data_cache_maml(self, data_path, test_task_index, add_dim_pos, ppn):

        datasets_cache = {'train': [], 'test': []}
        train_data_path = osp.join(data_path, 'train_data_embedding_%s.pkl' % str(ppn))
        test_data_path = osp.join(data_path, 'test_data_embedding_%s.pkl' % str(ppn))

        train_data = obj_unserialization(train_data_path)
        test_data = obj_unserialization(test_data_path)
        
        # print(test_task_index)
        for i, data_name in enumerate(TRAINING_TASK_SET):
            if test_task_index == data_name:
                test_task_index = i + 1
        # print(test_task_index)
        for i, key in enumerate(train_data.keys()):
            if i == test_task_index - 1:
                test_spt_x, test_spt_y, _ = self.split_x_y(train_data[key], add_dim_pos, ppn=ppn)
                test_qry_x, test_qry_y, _ = self.split_x_y(test_data[key], add_dim_pos, ppn=ppn)
                datasets_cache['test'].append([test_spt_x, test_spt_y, test_qry_x, test_qry_y])
                self.task_id['test'].append(key)
                continue
            
            train_spt_x, train_spt_y, _ = self.split_x_y(train_data[key], add_dim_pos, ppn=ppn)
            train_qry_x, train_qry_y, _ = self.split_x_y(test_data[key],  add_dim_pos,  ppn=ppn)
            datasets_cache['train'].append([train_spt_x, train_spt_y, train_qry_x, train_qry_y])
            self.task_id['train'].append(key)
        
        self.task_num = len(datasets_cache['train'])

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
        # print('split train & val:')
        # print('train: %d, %d' % (len(spt[0]), len(spt[0][0])))
        # print('val: %d, %d' % (len(qry[0]), len(qry[0][0])))
        return spt, qry

    def split_x_y(self, data, add_dim_pos, ppn=10):
        
        forecast_point_num = ppn
        position = len(data[0]) - forecast_point_num
        xs = np.array(data)[:, :position]
        ys = np.array(data)[:, position:]
        # ==================================================== #
        # update time: 2021-12-10
        if add_dim_pos == -1:
            np_xs = np.array(xs)
        else:
            np_xs = from_numpy(xs).unsqueeze(dim=1).numpy()
        self.features = len(xs[0])
        # ==================================================== #
        self.output = forecast_point_num
        self.outputs.append(forecast_point_num)
        # print('split x & y')
        # print(xs.shape)
        # print(ys.shape)

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