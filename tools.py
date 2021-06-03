# -*- coding:utf-8 -*-
__author__ = 'XF'

'''
The script is set for supplying some tool function.
'''

import os
import time
import pickle
import pandas as pds



def read_tsv(path=None, header=None):

    if path is None:
        raise FileExistsError('The path is None!')

    content = pds.read_csv(path, sep='\t', header=header, )
    return content


# object serialization
def obj_serialization(path, obj):

    if obj is not None:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        print('object is None!')


# object instantiation
def obj_unserialization(path):

    if os.path.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)


def generate_filename(suffix, *args, sep='_', timestamp=False):

    '''

    :param suffix: suffix of file
    :param sep: separator，default '_'
    :param timestamp: add timestamp for uniqueness
    :param args:
    :return:
    '''

    filename = sep.join(args).replace(' ', '_')
    if timestamp:
        filename += time.strftime('_%Y%m%d%H%M%S')
    if suffix[0] == '.':
        filename += suffix
    else:
        filename += ('.' + suffix)

    return filename


def metrics(y, y_hat):
    assert y.shape == y_hat.shape  # Tensor y and Tensor y_hat must have the same shape
    y = y.cpu()
    y_hat = y_hat.cpu()

    # smape
    _smape = smape(y, y_hat)

    # rmse
    _rmse = rmse(y, y_hat)

    return _rmse, _smape


def smape(Y, Y_hat):
    temp = [abs(y - y_hat) / (abs(y) + abs(y_hat)) for y, y_hat in zip(Y.view(-1).numpy(), Y_hat.view(-1).numpy())]
    return (sum(temp) / len(temp)) * 200


def rmse(Y, Y_hat):

    temp = [pow(y - y_hat, 2) for y, y_hat in zip(Y.view(-1).numpy(), Y_hat.view(-1).numpy())]
    return pow(sum(temp) / len(temp), 0.5)