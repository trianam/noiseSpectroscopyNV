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
        self.convLayers = len(conf.filters)
        self.fcLayers = len(conf.hiddenDim)

        self.cnn = nn.ModuleList()
        self.mp = nn.ModuleList()
        outFilters = 1
        convDim = conf.dimX
        for i in range(self.convLayers):
            self.cnn.append(nn.Conv1d(outFilters, conf.filters[i], conf.convKernel[i]))
            outFilters = conf.filters[i]

            if conf.pooling == 'max':
                self.mp.append(nn.MaxPool1d(conf.poolKernel[i]))
            elif conf.pooling == 'avg':
                self.mp.append(nn.AvgPool1d(conf.poolKernel[i]))
            else:
                raise ValueError("Pooling {} not valid".format(conf.pooling))

            convDim = (convDim - conf.convKernel[i] + 1 - conf.poolKernel[i]) / conf.poolKernel[i] +1

        if conf.dropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

        self.fc = nn.ModuleList()
        outDim = int(convDim)*conf.filters[-1]+1
        for i in range(self.fcLayers):
            self.fc.append(nn.Linear(outDim, conf.hiddenDim[i]))
            outDim = conf.hiddenDim[i]
        self.fc.append(nn.Linear(outDim, conf.dimY))

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
        b = true['b']

        x = x.view(x.shape[0], 1, x.shape[1])

        for i in range(self.convLayers):
            x = self.cnn[i](x)
            x = self._activation(x, self.conf.activation)
            x = self.mp[i](x)
            if self.conf.dropout > 0:
                x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = torch.cat((b, x), 1)

        for i in range(self.fcLayers):
            x = self.fc[i](x)
            x = self._activation(x, self.conf.activation)

        x = self.fc[-1](x)
        x = self._activation(x, self.conf.finalActivation)

        return {'y':x}
