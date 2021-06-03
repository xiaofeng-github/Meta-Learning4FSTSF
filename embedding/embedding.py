# -*- coding:utf-8 -*-
__author__ = 'XF'
__date__ = '2021-05-05'
'''
This script is set to finish time series data embedding for uniting the length of data.
'''

# builtins library
import os
from collections import OrderedDict

# third-party library
import torch
import numpy as np
import torch.nn as nn

# self-defined wheels
from data_preprocessing import normalizer
from tools import obj_serialization, obj_unserialization
from configs import DATADIR

class EmbeddingBiGRU(nn.Module):

    def __init__(self, n_input, n_hidden, batch_size=100, bidirectional=True, forecasting_point_num=10):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.forecasting_point_num = forecasting_point_num

        self.bigru = nn.GRU(
            input_size=self.n_input,
            hidden_size=self.n_hidden,
            bidirectional=self.bidirectional,
            num_layers=1,

        )

    def batch_train(self, x):

        # x: (record size, seq_length, dim feature)
        record_size = x.shape[0]
        batch_data = []
        if record_size <= self.batch_size:
            batch_data.append(x)
        else:
            for pos in range(0, record_size, self.batch_size):
                batch_data.append(x[pos:pos + self.batch_size, :])
            if pos + self.batch_size < record_size:
                batch_data.append(x[pos + self.batch_size:, :])
        return batch_data

    def forward(self, x):

        # x: (batch size, seq_length, dim_feature) --> (seq_length, batch size, input size)
        batch_data = self.batch_train(x[:, :x.shape[1] - self.forecasting_point_num, :])
        forecasting_data = x[:, x.shape[1] - self.forecasting_point_num:, :].squeeze(dim=2).numpy()
        embedding = []
        for batch in batch_data:
            batch = batch.contiguous().view(batch.shape[1], len(batch), -1)
            gru_out, h_n = self.bigru(batch)

            forward_embedding = h_n[0, :, :].detach().numpy()
            backward_embedding = h_n[1, :, :].detach().numpy()
            embedding.append(np.concatenate((forward_embedding, backward_embedding), axis=1))
        embedding = np.concatenate(embedding, axis=0)
        embedding = np.concatenate((embedding, forecasting_data), axis=1)
        return embedding.astype(np.float64)

    pass


if __name__ == '__main__':

    train_data_path = os.path.join(DATADIR, 'few_shot_data\\train_data.pkl')
    test_data_path = os.path.join(DATADIR, 'few_shot_data\\test_data.pkl')
    forecasting_point_num = 40
    model = EmbeddingBiGRU(n_input=1, n_hidden=100, forecasting_point_num=forecasting_point_num)

    # train data embedding ……
    print('train data embedding .......')
    train_data = obj_unserialization(train_data_path)
    train_data_embedding = OrderedDict()

    for key, value in train_data.items():
        input_data = torch.from_numpy(normalizer(value)).float().unsqueeze(dim=2)
        embedding_data = model(input_data)
        train_data_embedding.setdefault(key, embedding_data)
    print('train data dimension: %d' % len(train_data_embedding['0001'][0]))
    obj_serialization(os.path.join(DATADIR, 'few_shot_data\\train_data_embedding_%s.pkl' % str(forecasting_point_num)), train_data_embedding)

    # test data embedding ……
    print('test data embedding ......')
    test_data = obj_unserialization(test_data_path)
    test_data_embedding = OrderedDict()

    for key, value in test_data.items():
        input_data = torch.from_numpy(normalizer(value)).float().unsqueeze(dim=2)
        embedding_data = model(input_data)
        test_data_embedding.setdefault(key, embedding_data)
    print('test data dimension: %d' % len(test_data_embedding['0001'][0]))
    obj_serialization(os.path.join(DATADIR, 'few_shot_data\\test_data_embedding_%s.pkl' % str(forecasting_point_num)), test_data_embedding)

    print('OK!')
    pass
