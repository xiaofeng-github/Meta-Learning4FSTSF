# -*- coding:utf-8 -*-
__author__ = 'XF'

'default configuration for this project'

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# configuration for data_preprocessing.py
# data dir
DATADIR = os.path.join(BASE_DIR, 'data')
# few-shot dataset name
few_shot_dataset_name = [
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'Car',
    'Coffee',
    'FaceFour',
    'Herring',
    'Lightning2',
    'Lightning7',
    'Meat',
    'OliveOil',
    'Rock',
    'Wine'
]

# configuration for options.py
model_save_dir = os.path.join(BASE_DIR, 'model')
loss_save_dir = os.path.join(BASE_DIR, 'loss')
log_save_path = os.path.join(BASE_DIR, 'neural network\\logs.txt')
data_dir = os.path.join(BASE_DIR, '%s\\trainingData' % '')
tensorBoardX_dir = os.path.join(BASE_DIR, 'loss\\tensorBoardX')
console_file = os.path.join(BASE_DIR, 'log.txt')
exp_result_dir = os.path.join(BASE_DIR, 'neural network/maml/exp_result/batch_task_15')


MODEL_NAME = ['lstm', 'cnn', 'cnnConlstm', 'lstm+maml', 'cnn+maml', 'cnnConlstm+maml']
MODE_NAME = ['training', 'testing', 'together']


if __name__ == '__main__':

    pass
