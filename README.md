# Meta-Learning4FSTSF
Meta-Learning for Few-Shot Time Series Forecasting

# Usage 

This section of the README walks through how to train the models.

## data prepare
> data_preprocessing.py + embedding.py

**notes**: 
The time-series data given in '/data/few_shot_data/...' already have done this step. For new raw time-series data, the two scripts can be used in this step.


## training of Base_{model}
### In this phase, a dataset is a time-series task, and each task would be training seperately.

>**main.py**
>>**Arguments help:**

    --baseNet: [mlp/cnn/lstm/cnnConlstm]
    --dataset: the directory of saving pre-processed time-series data 
    --update_step_target: update times of network
    --fine_lr: leanring rate in this phase
    --ppn: predict point number [10/20/30/40]
    --device: [cpu/cuda]
    --user_id: the name of the task that will be training, it can be found in ./config.py TRAINING_TASK_SET

>**training single task:**

'''
python main.py --baseNet [mlp/cnn/lstm/cnnConlstm] --dateset [few_shot_data/your defined data dir] --update_step_target 10 --fine_lr 0.001 --ppn [10/20/30/40] --device [cpu/cuda] --user_id 0001
'''

>**training all task:**

'''
python main.py --baseNet [mlp/cnn/lstm/cnnConlstm] --dateset few_shot_data --update_step_target 10 --fine_lr 0.001 --ppn [10/20/30/40] --device [cpu/cuda]
'''


## training Meta_{model}
### In this phase, one task is selected as target task, and the remains are training-task set, firstly training baseNet using support set of training-task set, and then training MetaNet using query set of training-task set, finally using support set of target task to fine tune MetaNet. 

>**main.py**
>>**Argument help:**

    --maml: using 'maml mode' to training model
    --update_step_train: the update times of baseNet on training-task set
    --update_step_target: the update times of MetaNet on target task
    --epoch: iteration times
    --base_lr: the learning rate of baseNet
    --meta_lr: the learning rate of MetaNet
    --fine_lr: the learning rate of MetaNet during fine-tuing

>**training single task:**

'''
python main.py --baseNet [cnn/lstm/cnnConlstm] --maml --dataset few_shot_data --epoch 10 --update_step_train 10 --update_step_target 10 --base_lr 0.01 --meta_lr 0.01 --fine_lr 0.01 --ppn 10 --device [cpu/cuda] --user_id Wine
'''

>**training all task:**

'''
python main.py --baseNet [cnn/lstm/cnnConlstm] --maml --dataset few_shot_data --epoch 10 --update_step_train 10 --update_step_target 10 --base_lr 0.01 --meta_lr 0.01 --fine_lr 0.01 --ppn 10 --device [cpu/cuda]
'''

## results
### All the trained models and evaluating metrics would be saved in dir ./results/

## log
## Some useful log information would be saved in dir ./log/