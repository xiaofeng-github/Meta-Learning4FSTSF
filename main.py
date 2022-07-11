# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2022/03/10'

'begin from here'


# built-in library
import os.path as osp
import time

# third-party library

# self-defined tools
from core.options import parse_args
from core.train import train
from configs import TRAINING_TASK_SET, DATA_DIR

if __name__ == '__main__':

    # python train_maml.py --mode together --model cnn+maml --epoch 100  --dataSet maml_load_data(14).pkl --ratio 0.9 --time_size 3 --update_step 10 --update_step_test 20 --meta_lr 0.001 --base_lr 0.01

    print('--------------------------------------------Time Series Forecasting------------------------------------------')
    params = parse_args('main')
    start = time.time()
    if params.user_id == 'none':
        # conducting all tasks
        for user_id in TRAINING_TASK_SET:
            train(
                epoch_num=params.epoch,
                test_user_index=user_id,
                add_dim_pos=params.add_dim_pos,
                data_path=osp.join(DATA_DIR, params.dataset),
                update_step=params.update_step,
                update_step_test=params.update_step_test,
                meta_lr=params.meta_lr,
                base_lr=params.base_lr,
                fine_lr=params.fine_lr,
                device=params.device,
                baseNet=params.baseNet,
                maml=params.maml,
                log=params.log,
                maml_log=params.maml_log,
                lsd=params.lsd,
                ppn=params.ppn,
                batch_task_num=params.batch_task_num,
                rmse_path=params.rmse_path,
                mape_path=params.mape_path,
                smape_path=params.smape_path,
                ft_step=params.ft_step,
                new_settings=params.new_settings
            )
        end = time.time()
        print('using time: %.4f Hour' % ((end - start) / 3600.0))
        print('training is over!')
    elif params.user_id in TRAINING_TASK_SET:
        # conducting single task
        smape, rmse = train(
                    epoch_num=params.epoch,
                    test_user_index=params.user_id,
                    add_dim_pos=params.add_dim_pos,
                    data_path=osp.join(DATA_DIR, params.dataset),
                    update_step=params.update_step,
                    update_step_test=params.update_step_test,
                    meta_lr=params.meta_lr,
                    base_lr=params.base_lr,
                    fine_lr=params.fine_lr,
                    device=params.device,
                    baseNet=params.baseNet,
                    maml=params.maml,
                    log=params.log,
                    maml_log=params.maml_log,
                    lsd=params.lsd,
                    ppn=params.ppn,
                    rmse_path=params.rmse_path,
                    mape_path=params.mape_path,
                    smape_path=params.smape_path,
                    ft_step=params.ft_step,
                    new_settings=params.new_settings
                )
        end = time.time()
        print('using time: %.4f Min' % ((end - start) / 60.0))
        print('training is over!')
        print('smape: %.4f' % smape)
        print('rmse: %.4f' % rmse)
    else:
        raise Exception('Unknown user id!')

 
