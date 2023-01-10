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

        if conf.rnnType == 'lstm':
            self.rnn = nn.LSTM(input_size=conf.dimX, hidden_size=conf.hiddenDim, num_layers=conf.hiddenLayers, batch_first=True, dropout=conf.dropout)
        elif conf.rnnType == 'gru':
            self.rnn = nn.GRU(input_size=conf.dimX, hidden_size=conf.hiddenDim, num_layers=conf.hiddenLayers, batch_first=True, dropout=conf.dropout)
        else:
            raise ValueError("rnnType {} not valid".format(conf.rnnType))

        self.fcOut = nn.Linear(conf.hiddenDim, conf.dimY)

        if conf.inputDropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

        self.conf = conf

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

        if self.conf.inputDropout > 0:
            x = self.dropout(x)

        x,_ = self.rnn(x)
        x = x[:,-1,:]

        x = self.fcOut(x)
        x = self._activation(x, self.conf.finalActivation)

        return {'y':x}
