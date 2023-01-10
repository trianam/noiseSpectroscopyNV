#     loaders.py
#     The class and function to create the Pytorch dataset and dataloader.
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
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import h5py
import os

class DatasetNpz(torch.utils.data.Dataset):
    def __init__(self, conf, mySet=None):
        minMax = (  #from data creation
            (0.0005, 0.01),
            (0.1, 1),
            (50, 1000),
            (0.001, 0.01),
        )

        self.conf = conf

        if 'data' in os.listdir('.'):
            basePath = "data"
        else:
            basePath = "../../../data" #go up for tune

        if conf.customValidTest is None:
            fileDataset = np.load(os.path.join(basePath,conf.dataset))
        else:
            fileDataset = np.load(os.path.join(basePath, conf.customValidTest[mySet]))

        self.data = {}

        x = fileDataset['coh']
        if fileDataset['par'].shape[1] == 5:
            y = fileDataset['par'][:,:-1] #skip nN
        else:
            y = fileDataset['par']
        # y = fileDataset['par'][:,-2] #skip nN
        # y = y.reshape(-1,1)

        if conf.normalizeY:
            for i in range(y.shape[1]):
                y[:,i] -= minMax[i][0]
                y[:,i] /= minMax[i][1] - minMax[i][0]
            # y[:,0] -= minMax[3][0]
            # y[:,0] /= minMax[3][1] - minMax[3][0]

        if not conf.split is None and conf.customValidTest is None:
            fileSplit = np.load(os.path.join(basePath,conf.split))
            if not mySet is None:
                split = fileSplit[mySet]
            else:
                split = [i for s in [fileSplit[k] for k in fileSplit] for i in s]

            self.data['x'] = x[split]
            self.data['y'] = y[split]
        else:
            self.data['x'] = x
            self.data['y'] = y

        if conf.inputB:
            self.data['b'] = self.data['y'][:,2]
            self.data['y'] = self.data['y'][:,[0,1,3]]

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        #x = torch.from_numpy(self.x[idx])
        # x = torch.Tensor(self.data['x'][idx])
        # y = torch.Tensor(self.data['y'][idx])

        toReturn = {
            'x': self.data['x'][idx],
            'y': self.data['y'][idx],
        }

        if self.conf.inputB:
            b = self.data['b'][idx]
            toReturn['b'] = b.reshape(1)

        return toReturn

def npz(conf, all=False, customSet=None):
    if not customSet is None:
        dataset = DatasetNpz(conf, customSet)
        loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batchSize, shuffle=conf.shuffleDataset)

        return loader, dataset

    if all:
        datasets = DatasetNpz(conf)
        loaders = torch.utils.data.DataLoader(datasets, batch_size=conf.batchSize, shuffle=conf.shuffleDataset)
    else:
        if conf.split is None:
            datasets = {'all': DatasetNpz(conf)}
            loaders = {'all': torch.utils.data.DataLoader(datasets['all'], batch_size=conf.batchSize, shuffle=conf.shuffleDataset)}
        else:
            splits = ['train', 'valid', 'test']

            datasets = {s: DatasetNpz(conf, s) for s in splits}
            loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, shuffle=conf.shuffleDataset) for s in splits}
    
    return loaders, datasets





class DatasetHdf5Test(torch.utils.data.Dataset):
    def __init__(self, conf, split=None):
        self.conf = conf

        if 'data' in os.listdir('.'):
            basePath = "data"
        else:
            basePath = "../../../data" #go up for tune

        self.fileDataset = h5py.File(os.path.join(basePath,conf.dataset), 'r')

        self.data = {}

        self.data['x'] = self.fileDataset['coh']
        if conf.normalizeY:
            self.data['y'] = self.fileDataset['parNorm']
        else:
            self.data['y'] = self.fileDataset['par']

        if not split is None:
            # self.fileSplit = h5py.File(os.path.join(basePath,conf.split), 'r')
            # self.split = self.fileSplit[split][:]
            with h5py.File(os.path.join(basePath,conf.split), 'r') as fileSplit:
                self.split = fileSplit[split][()]
        else:
            self.split = None

    def __len__(self):
        if not self.split is None:
            return len(self.split)
        else:
            return len(self.data['y'])

    def __getitem__(self, idx):
        #x = torch.from_numpy(self.x[idx])
        # x = torch.Tensor(self.data['x'][idx])
        # y = torch.Tensor(self.data['y'][idx])

        if not self.split is None:
            x = self.data['x'][self.split[idx]]
            y = self.data['y'][self.split[idx]]
        else:
            x = self.data['x'][idx]
            y = self.data['y'][idx]

        return {'x': x, 'y': y}

    def close(self):
        self.fileDataset.close()
        # if not self.split is None:
        #     self.fileSplit.close()


def hdf5Test(conf):
    numWorkers = 8#100
    if 'data' in os.listdir('.'):
        basePath = "data"
    else:
        basePath = "../../../data" #go up for tune

    if conf.split is None:
        datasets = {'all': DatasetHdf5(conf)}
        loaders = {'all': torch.utils.data.DataLoader(datasets['all'], batch_size=conf.batchSize, shuffle=conf.shuffleDataset, num_workers=numWorkers)}
    else:
        with h5py.File(os.path.join(basePath,conf.split), 'r') as fileSplit:
            splits = list(fileSplit.keys())
        datasets = {s: DatasetHdf5(conf, s) for s in splits}
        loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, num_workers=numWorkers) for s in splits}

    return loaders, datasets




class DatasetHdf5(torch.utils.data.Dataset):
    def __init__(self, conf, splitSet):
        self.conf = conf

        if 'data' in os.listdir('.'):
            basePath = "data"
        else:
            basePath = "../../../data" #go up for tune

        if conf.customValidTest is None:
            self.fileDataset = h5py.File(os.path.join(basePath,conf.dataset), 'r')
        else:
            self.fileDataset = h5py.File(os.path.join(basePath,conf.customValidTest[splitSet]), 'r')

        self.data = {}

        self.data['x'] = self.fileDataset['coh']
        if conf.normalizeY:
            self.data['y'] = self.fileDataset['parNorm']
        else:
            self.data['y'] = self.fileDataset['par']

        if conf.inputB:
            self.data['b'] = self.data['y'][:,2]
            self.data['y'] = self.data['y'][:,[0,1,3]]

        if conf.split is None or not conf.customValidTest is None:
            self.minIdx = 0
            self.maxIdx = len(self.data['y'])
        else:
            dataLen = len(self.data['y'])
            testLen = int(dataLen / 100. * conf.split)

            if splitSet == 'train':
                self.minIdx = 0
                self.maxIdx = dataLen - (2 * testLen)
            elif splitSet == 'valid':
                self.minIdx = dataLen - (2 * testLen)
                self.maxIdx = dataLen - testLen
            elif splitSet == 'test':
                self.minIdx = dataLen - testLen
                self.maxIdx = dataLen

    def __len__(self):
        # return len(self.data['y'])
        return self.maxIdx - self.minIdx

    def __getitem__(self, idx):
        #x = torch.from_numpy(self.x[idx])
        # x = torch.Tensor(self.data['x'][idx])
        # y = torch.Tensor(self.data['y'][idx])

        # x = self.data['x'][idx]
        # y = self.data['y'][idx]

        x = self.data['x'][self.minIdx + idx]
        y = self.data['y'][self.minIdx + idx]
        toReturn = {'x': x, 'y': y}

        if self.conf.inputB:
            b = self.data['b'][self.minIdx + idx]
            toReturn['b'] = b.reshape(1)

        return toReturn


    def close(self):
        self.fileDataset.close()

class DatasetHdf5Alt(torch.utils.data.Dataset):
    def __init__(self, conf):
        self.conf = conf

        if 'data' in os.listdir('.'):
            basePath = "data"
        else:
            basePath = "../../../data" #go up for tune

        self.fileDataset = os.path.join(basePath,conf.dataset)

        self.kX = 'coh'
        if conf.normalizeY:
            self.kY = 'parNorm'
        else:
            self.kY = 'par'

    def __len__(self):
        with h5py.File(self.fileDataset, 'r') as f:
            dataLen = len(f[self.kY])

        return dataLen

    def __getitem__(self, idx):
        with h5py.File(self.fileDataset, 'r') as f:
            x = f[self.kX][idx]
            y = f[self.kY][idx]
            # x = torch.Tensor(f[self.kX][idx])
            # y = torch.Tensor(f[self.kY][idx])

        return {'x': x, 'y': y}

def hdf5(conf, customSet=None):
    if conf.debug:
        numWorkers = 0 #to debug
    else:
        numWorkers = 8#100

    if 'data' in os.listdir('.'):
        basePath = "data"
    else:
        basePath = "../../../data" #go up for tune

    if not customSet is None:
        dataset = DatasetHdf5(conf, customSet)
        loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batchSize, num_workers=numWorkers)

        return loader, dataset
    else:
        if conf.split is None:
            datasets = {'all': DatasetHdf5(conf)}
            loaders = {'all': torch.utils.data.DataLoader(datasets['all'], batch_size=conf.batchSize, shuffle=conf.shuffleDataset, num_workers=numWorkers)}
        else:
            # fileSplit = h5py.File(os.path.join(basePath,conf.split), 'r')
            # samplers = {s: torch.utils.data.sampler.SubsetRandomSampler(fileSplit[s]) for s in fileSplit}
            # loaders = {s: torch.utils.data.DataLoader(dataset, batch_size=conf.batchSize, num_workers=numWorkers, sampler=samplers[s]) for s in fileSplit}

            splits = ['train', 'valid', 'test']
            datasets = {s: DatasetHdf5(conf, s) for s in splits}
            loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, num_workers=numWorkers) for s in splits}

        return loaders, datasets




class DatasetMultiCollapse(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.data = {}
        self.data['x'] = x
        self.data['y'] = y

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        toReturn = {'x': x, 'y': y}
        return toReturn

def multiCollapse(conf, all=False, customSet=None):
    splits = ['train', 'valid', 'test']

    minMax = conf.multiCollapseMinMax

    if 'data' in os.listdir('.'):
        basePath = "data"
    else:
        basePath = "../../../data" #go up for tune

    fileDataset = np.load(os.path.join(basePath,conf.dataset))

    x = fileDataset['coh']
    y = fileDataset['par']

    if len(y.shape) == 3:
        y = y[:,0,:]    # equal for all nN

    allCollapseRanges = conf.multiCollapseRanges

    # Why?
    # collapseRange = (
    #     min(allCollapseRanges[conf.rangeCollapse[0]] + allCollapseRanges[conf.rangeCollapse[1]-1]),
    #     max(allCollapseRanges[conf.rangeCollapse[0]] + allCollapseRanges[conf.rangeCollapse[1]-1]),
    # )

    collapseRange = (allCollapseRanges[conf.rangeCollapse[0]][0], allCollapseRanges[conf.rangeCollapse[1]-1][1])

    x = x[:, conf.rangeNN[0]:conf.rangeNN[1], collapseRange[0]:collapseRange[1]]

    # x = x.reshape(x.shape[0], -1)
    x = x.swapaxes(1,2)

    if conf.normalizeY:
        for i in range(y.shape[1]):
            y[:,i] -= minMax[i][0]
            y[:,i] /= minMax[i][1] - minMax[i][0]
        # y[:,0] -= minMax[3][0]
        # y[:,0] /= minMax[3][1] - minMax[3][0]

    if conf.multiCollapseIgnoreB:
        y = y[:,[0,1,3]]

    fileSplit = np.load(os.path.join(basePath,conf.split))

    x = {s: x[fileSplit[s]] for s in splits}
    y = {s: y[fileSplit[s]] for s in splits}

    datasets = {s: DatasetMultiCollapse(x[s], y[s]) for s in splits}
    loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, shuffle=False) for s in splits}

    return loaders, datasets


def multiCollapseExperimental(conf):

    minMax = conf.multiCollapseMinMax

    if 'data' in os.listdir('.'):
        basePath = "data"
    else:
        basePath = "../../../data" #go up for tune

    fileDataset = np.load(os.path.join(basePath,conf.datasetExperimental))

    x = fileDataset['coh']
    y = fileDataset['par']

    if len(y.shape) == 3:
        y = y[:,0,:]    # equal for all nN

    allCollapseRanges = conf.multiCollapseRanges

    collapseRange = (allCollapseRanges[conf.rangeCollapse[0]][0], allCollapseRanges[conf.rangeCollapse[1]-1][1])

    x = x[:, conf.rangeNN[0]:conf.rangeNN[1], collapseRange[0]:collapseRange[1]]

    # x = x.reshape(x.shape[0], -1)
    x = x.swapaxes(1,2)

    if conf.normalizeY:
        for i in range(y.shape[1]):
            y[:,i] -= minMax[i][0]
            y[:,i] /= minMax[i][1] - minMax[i][0]
        # y[:,0] -= minMax[3][0]
        # y[:,0] /= minMax[3][1] - minMax[3][0]

    if conf.multiCollapseIgnoreB:
        y = y[:,[0,1,3]]

    dataset = DatasetMultiCollapse(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batchSize, shuffle=False)

    return loader, dataset
