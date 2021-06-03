# Meta-Learning4FSTSF
Meta-Learning for Few-Shot Time Series Forecasting

## time series embedding
> data_preprocessing.py + embedding.py

## training of Base_{model}
> train.py
> > train.py --model cnn --baseNet cnn --ppn 10 --update_step_test 500 --fine_lr 0.5 --epoch 119

## training and testing Meta_{model}
> train.py
> > train.py --model cnn+maml --baseNet cnn --maml --ppn 10 --update_step 5  --update_step_test 500 --base_lr 0.1 --meta_lr 0.5 --fine_lr 0.5 --epoch 47 --begin_task 1 --end_task 119