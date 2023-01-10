#!/usr/bin/env python

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import configurations
import funPytorch as fun
import notifier
from datetime import datetime
import numpy as np


def _funcGauss(x,y0, a,xc,w):
    return y0+a*np.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

def _funcNoise(x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
    return y0 + _funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

def loss(yhatBatch,yBatch):
    # Y0, A, B, W1

    B_lims   = [520,536]
    W1_lims   = [0.004,0.009]

    g=1.0705e-3 # C-13 nuclear spin gyromagnetic ratio
    omega2 = 2*np.pi*np.linspace(g*min(B_lims)-5*max(W1_lims), g*max(B_lims)+5*max(W1_lims),500)


    error = 0
    for yhat,y in zip(yhatBatch, yBatch):
        vl = yhat[2]*g # B*\gamma [MHZ]
        para_A = [0.0, yhat[1], vl, yhat[3]] # [offset, amplitude, center, width] All in MHz

        vl = y[2]*g # B*\gamma [MHZ]
        para_B = [0.0, y[1], vl, y[3]] # [offset, amplitude, center, width] All in MHz

        error += abs(yhat[0] - y[0])*(8.5-0.001) + sum(abs(_funcNoise(omega2,*para_B)-_funcNoise(omega2,*para_A)))*(omega2[1]-omega2[0])

    return error / len(yBatch)

if __name__ == '__main__':
    timeFormat = "%Y/%m/%d - %H:%M:%S"

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Use {} configName [gpuNum (def. 0)]".format(sys.argv[0]))
    else:
        if len(sys.argv) == 3:
            device = "cuda:{}".format(sys.argv[2])
        else:
            device = "cuda:0"
        # conf = getattr(sys.modules['configurations'], sys.argv[1])
        conf = eval('configurations.{}'.format(sys.argv[1]))

        conf.runningPredictions = True
        conf.nonVerbose = False
        # conf = conf.copy({"runningPredictions": True})

        startTime = datetime.now()

        print("====================")
        print("RUN USING {} on device {}".format(sys.argv[1], device))
        print(startTime.strftime(timeFormat))
        print("====================")
        print("======= LOAD MODEL")
        model,optim,loadEpoch,_ = fun.loadModel(conf, device)
        print("======= LOAD DATA")
        dataloaders, _ = fun.processData(conf)
        print("======= CALCULATE PREDICTIONS")
        pred = fun.predict(conf, model, dataloaders, loadEpoch, toSave=False, toReturn=True)

        print("======= SLOSS:")
        print(loss(pred['test']['pred'], pred['test']['y']))

        endTime = datetime.now()
        print("=======")
        print(endTime.strftime(timeFormat))
        print("====================")
        # notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),endTime.strftime(timeFormat),str(endTime-startTime).split('.', 2)[0]))

