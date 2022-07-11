# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2022/03/10'

'default configuration for this project'

import os.path as osp


BASE_DIR = osp.dirname(osp.abspath(__file__))

dataName = 'data'
model_save_dir = osp.join(BASE_DIR, 'model')
loss_save_dir = osp.join(BASE_DIR, 'results/loss')
log_save_path = osp.join(BASE_DIR, 'log/logs.txt')
log_save_dir = osp.join(BASE_DIR, 'log')
DATA_DIR = osp.join(BASE_DIR, 'data')
MODEL_PATH = osp.join(BASE_DIR, 'results/model')
console_file = osp.join(BASE_DIR, 'log/logs.txt')
exp_result_dir = osp.join(BASE_DIR, 'results')

MODEL_NAME = ['mlp', 'lstm', 'cnn', 'cnnConlstm', 'lstm+maml', 'cnn+maml', 'cnnConlstm+maml']
MODE_NAME = ['training', 'testing', 'together']

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

TRAINING_TASK_SET = [
    '0001',
    '0002',
    '0003',
    '0004',
    '0005',
    '0006',
    '0007',
    '0008',
    '0009',
    '0010',
    '0011',
    '0012',
    '0013',
    '0014',
    '0015',
    '0016',
    '0022',
    '0023',
    '0024',
    '0025',
    '0026',
    '0029',
    '0030',
    '0031',
    '0032',
    '0037',
    '0046',
    '0047',
    '0048',
    '0049',
    '0050',
    '0051',
    '0054',
    '0055',
    '0056',
    '0066',
    '0069',
    '0070',
    '0071',
    '0082',
    '0085',
    '0088',
    '0089',
    '0090',
    '0091',
    '0092',
    '0093',
    '0094',
    '0095',
    '0096',
    '0097',
    '0098',
    '0099',
    '0100',
    '0102',
    '0103',
    '0104',
    '0106',
    '0107',
    '0108',
    '0110',
    '0111',
    '0112',
    '0113',
    '0114',
    '0115',
    '0116',
    '0118',
    '0119',
    '0120',
    '0121',
    '0122',
    '0123',
    '0124',
    '0125',
    '0126',
    '0127',
    '0128',
    '0129',
    '0130',
    '0131',
    '0132',
    '0133',
    '0134',
    '0135',
    '0136',
    '0137',
    '0138',
    '0139',
    '0140',
    '0141',
    '0142',
    '0143',
    '0144',
    '0145',
    '0146',
    '0147',
    '0148',
    '0149',
    '0150',
    '0151',
    '0152',
    '0153',
    '0154',
    '0155',
    '0156',
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

if __name__ == '__main__':

    print(BASE_DIR)
    pass
