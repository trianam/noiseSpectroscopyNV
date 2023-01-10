#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import sys
import glob
import numpy as np
import h5py
from datetime import datetime

basePath = 'data'

configurations = {
    'dataset32': {
        'inputFileName': "dataset32.npz",
        'outputFileName': "dataset32-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': True,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32-TEST': {
        'inputFileName': "dataset32-TEST.npz",
        'outputFileName': "dataset32-TEST-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': True,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32-TEST2': {
        'inputFileName': "dataset32-TEST2.npz",
        'outputFileName': "dataset32-TEST2-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': True,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32AugV2': {
        'inputFileName': "dataset32AugV2.hdf5",
        'outputFileName': "dataset32AugV2-split.hdf5",
        'copyData': False,
        'group': 10,
        'shuffleBefore': True,
        'shuffleAfter': True,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MC': {
        'inputFileName': "dataset32MC.npz",
        'outputFileName': "dataset32MC-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': True,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MCnoB': {
        'inputFileName': "dataset32MCnoB.npz",
        'outputFileName': "dataset32MCnoB-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': True,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MCR': {
        'inputFileName': "dataset32MCR.npz",
        'outputFileName': "dataset32MCR-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': False,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MCRnew': {
        'inputFileName': "dataset32MCRnew.npz",
        'outputFileName': "dataset32MCRnew-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': False,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MCRnew1000': {
        'inputFileName': "dataset32MCRnew1000.npz",
        'outputFileName': "dataset32MCRnew1000-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': False,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
    'dataset32MCRnew10000': {
        'inputFileName': "dataset32MCRnew10000.npz",
        'outputFileName': "dataset32MCRnew10000-split.npz",
        'copyData': False,
        'group': 1,
        'shuffleBefore': False,
        'shuffleAfter': False,
        'customSplit': None,    #None or tuple of 3 couples
        'testValidPercent': 20,
    },
}

if len(sys.argv) < 2 or not sys.argv[1] in configurations:
        raise Exception("Use {} ? (with ? one of the available).".format(sys.argv[0]))

conf = configurations[sys.argv[1]]

fileType = conf['inputFileName'].split(".")[-1]
if fileType == "npz":
    dataset = np.load(os.path.join(basePath, conf['inputFileName']))
elif fileType == "hdf5":
    dataset = h5py.File(os.path.join(basePath, conf['inputFileName']), 'r')
else:
    raise ValueError("Input file name {} not valid".format(conf['inputFileName']))

dataLen = dataset['par'].shape[0]
print("Loaded dataset {} ({} samples)".format(os.path.join(basePath, conf['inputFileName']), dataLen))

if conf['shuffleBefore']:
    # index = np.random.permutation(dataLen)
    if dataLen % conf['group'] != 0:
        raise ValueError("Data len {} and group {} are not compatible.".format(dataLen, conf['group']))
    index = np.random.permutation(np.arange(dataLen).reshape(int(dataLen/conf['group']), conf['group'])).reshape(-1)
else:
    index = np.array(range(dataLen))

if conf['customSplit'] is None:
    testLen = int(dataLen / 100. * conf['testValidPercent'])
    test = index[-testLen:]
    valid = index[-2 * testLen: -testLen]
    train = index[: -2 * testLen]
else:
    cs = conf['customSplits']
    train = index[cs[0][0]:cs[0][1]]
    valid = index[cs[1][0]:cs[1][1]]
    test = index[cs[2][0]:cs[2][1]]

if conf['shuffleAfter']:
    np.random.shuffle(test)
    np.random.shuffle(valid)
    np.random.shuffle(train)

fileType = conf['outputFileName'].split(".")[-1]
if fileType == "npz":
    if conf['copyData']:
        raise NotImplementedError()
    else:
        np.savez_compressed(os.path.join(basePath, conf['outputFileName']), train=train, valid=valid, test=test)
elif fileType == "hdf5":
    if conf['copyData']:
        with h5py.File(os.path.join(basePath, conf['outputFileName']), 'w') as h5File:
            h5File['index/train'] = train
            h5File['index/valid'] = valid
            h5File['index/test'] = test

            startTime = datetime.now()
            print("Start {}".format(startTime))
            startTimeIter = datetime.now()
            totI = 0
            data = {}
            for s,ss in zip(['train','valid','test'], [train,valid,test]):
                data[s] = {}
                for k in dataset:
                    data[s][k] = h5File.create_dataset(os.path.join("data", s, k), (ss.shape[0], dataset[k].shape[1]), dtype=dataset[k].dtype)
                    for i in range(len(ss)):
                        if i % 1000 == 0:
                            endTimeIter = datetime.now()
                            expTime = str((endTimeIter-startTimeIter)/1000 * ((len(ss)*len(list(dataset.keys()))*3)-1 - totI)).split('.', 2)[0]
                            startTimeIter = endTimeIter
                            totI += 1
                            print("Copying {} {}, {}/{} (exp: {})              ".format(s, k, i, len(ss), expTime), end='\r')
                        data[s][k][i] = dataset[k][ss[i]]
                    print("Copying {} {}, {}/{} (exp: {})              ".format(s, k, i, len(ss), expTime))
            endTime = datetime.now()
            print("End {} (tot: {})".format(endTime, str(endTimeIter-startTimeIter).split('.', 2)[0]))


    else:
        with h5py.File(os.path.join(basePath, conf['outputFileName']), 'w') as h5File:
            h5File['train'] = train
            h5File['valid'] = valid
            h5File['test'] = test
else:
    raise ValueError("Output file name {} not valid".format(conf['outputFileName']))

print("Saved Split in {}".format(os.path.join(basePath, conf['outputFileName'])))
