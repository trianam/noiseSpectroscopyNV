#     modelMLP.py
#     The class for the MLP model.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()

        self.conf = conf

        self.cnn1 = nn.Conv1d(1, conf.filters, 43)
        self.cnn2 = nn.Conv1d(conf.filters, conf.filters, 44)
        self.cnn3 = nn.Conv1d(conf.filters, conf.filters, 43)
        if conf.pooling == 'max':
            self.mp1 = nn.MaxPool1d(4)
            self.mp2 = nn.MaxPool1d(4)
            self.mp3 = nn.MaxPool1d(4)
        elif conf.pooling == 'avg':
            self.mp1 = nn.AvgPool1d(4)
            self.mp2 = nn.AvgPool1d(4)
            self.mp3 = nn.AvgPool1d(4)
        else:
            raise ValueError("Pooling {} not valid".format(conf.pooling))

        if conf.dropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

        self.fc1 = nn.Linear(conf.filters, conf.filters)
        self.fc2 = nn.Linear(conf.filters, 4)

    def _activation(self, x, activation):
        if activation == 'sigmoid':
            #x = F.sigmoid(x)
            x = torch.sigmoid(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'relu':
            x = F.relu(x)
        elif not activation == 'none':
            raise Exception("value of activation not valid")

        return x

    def forward(self, true):
        x = true['x']
        x = x.view(x.shape[0], 1, x.shape[1])

        x = self.cnn1(x)
        x = self._activation(x, self.conf.activation)
        x = self.mp1(x)
        if self.conf.dropout > 0:
            x = self.dropout(x)

        x = self.cnn2(x)
        x = self._activation(x, self.conf.activation)
        x = self.mp2(x)
        if self.conf.dropout > 0:
            x = self.dropout(x)

        x = self.cnn3(x)
        x = self._activation(x, self.conf.activation)
        x = self.mp3(x)
        if self.conf.dropout > 0:
            x = self.dropout(x)

        x = x.view(x.shape[0], x.shape[1])

        x = self.fc1(x)
        x = self._activation(x, self.conf.activation)

        x = self.fc2(x)
        x = self._activation(x, self.conf.finalActivation)

        return {'y':x}
