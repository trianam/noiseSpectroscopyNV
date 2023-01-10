#!/usr/bin/env python

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket

import numpy as np

import configurations
import funPytorch as fun
import notifier
from datetime import datetime

if __name__ == '__main__':
    timeFormat = "%Y/%m/%d - %H:%M:%S"

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Use {} configName [gpuNum (def. 0)]".format(sys.argv[0]))
    else:
        if len(sys.argv) == 3:
            device = "cuda:{}".format(sys.argv[2])
        else:
            device = "cuda:0"

        confAll = eval('configurations.{}'.format(sys.argv[1]))


        values = np.zeros((len(confAll), len(confAll[0])))
        for n in range(len(confAll)):
            for c in range(len(confAll[n])):
                conf = confAll[n][c]
                model,optim,loadEpoch,_ = fun.loadModel(conf, device)
                dataloaders, _ = fun.processData(conf)
                predictions = fun.predict(conf, model, dataloaders, loadEpoch, toSave=False, toReturn=True)

                y = predictions['test']['y']
                pred = predictions['test']['pred']
                loss = fun.MyLoss(conf, device)

                values[n][c] = loss(pred, y).item()

        print(values)
