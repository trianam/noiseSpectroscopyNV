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

        nextDim = conf.dimX
        outputDim = conf.dimY

        self.fcH = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.fcH.append(nn.Linear(nextDim, conf.hiddenDim))
            nextDim = conf.hiddenDim

        self.fcOut = nn.Linear(nextDim, outputDim)

        if conf.dropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

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
        # x = x.reshape(x.shape[0], -1)
        x = x.view(x.shape[0], -1)

        for i in range(len(self.fcH)):
            x = self.fcH[i](x)
            x = self._activation(x, self.conf.activation)
            if self.conf.dropout > 0:
                x = self.dropout(x)

        x = self.fcOut(x)
        x = self._activation(x, self.conf.finalActivation)

        return {'y':x}
