#!/usr/bin/env python

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import configurations
import funPytorch as fun
import notifier
import loaders
from datetime import datetime

def run(confName, device="cuda:0", verbose=False):
    timeFormat = "%Y/%m/%d - %H:%M:%S"

    # conf = getattr(sys.modules['configurations'], sys.argv[1])
    conf = eval('configurations.{}'.format(confName))

    conf.runningPredictions = True
    # conf = conf.copy({"runningPredictions": True})

    startTime = datetime.now()

    if verbose:
        print("====================")
        print("RUN USING {} on device {}".format(sys.argv[1], device))
        print(startTime.strftime(timeFormat))
        print("====================")
        print("======= LOAD MODEL")
    model, optim, loadEpoch, _ = fun.loadModel(conf, device)
    if verbose:
        print("======= LOAD DATA")
    # dataloaders, _ = fun.processData(conf)
    dataloader, _ = loaders.multiCollapseExperimental(conf)

    if verbose:
        print("======= CALCULATE PREDICTIONS")
    pred = fun.predict(conf, model, {'exp': dataloader}, loadEpoch, toReturn=True)
    if verbose:
        print(pred['exp']['pred'])

    endTime = datetime.now()
    if verbose:
        print("=======")
        print(endTime.strftime(timeFormat))
        print("====================")
    # notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),endTime.strftime(timeFormat),str(endTime-startTime).split('.', 2)[0]))

    return pred['exp']['pred']

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Use {} configName [gpuNum (def. 0)]".format(sys.argv[0]))
    else:
        if len(sys.argv) == 3:
            device = "cuda:{}".format(sys.argv[2])
        else:
            device = "cuda:0"

        _ = run(sys.argv[1], device, verbose=True)
