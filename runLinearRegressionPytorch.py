#!/usr/bin/env python

import os
import sys
import socket
import configurations
import notifier
from datetime import datetime
import numpy as np
import torch
# from torch.autograd import Variable



def slossNP(yhatBatch,yBatch):
    def _funcGauss(x,y0, a,xc,w):
        return y0+a*np.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

    def _funcNoise(x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
        return y0 + _funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

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

def slossTorch(yhatBatch,yBatch, device):
    def _funcGauss(x,y0, a,xc,w):
        return y0+a*torch.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

    def _funcNoise(x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
        return y0 + _funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

    # Y0, A, B, W1

    B_lims   = [520,536]
    W1_lims   = [0.004,0.009]

    g=1.0705e-3 # C-13 nuclear spin gyromagnetic ratio
    omega2 = 2*np.pi*torch.linspace(g*min(B_lims)-5*max(W1_lims), g*max(B_lims)+5*max(W1_lims),500).to(device)


    error = 0
    for yhat,y in zip(yhatBatch, yBatch):
        vl = yhat[2]*g # B*\gamma [MHZ]
        para_A = [0.0, yhat[1], vl, yhat[3]] # [offset, amplitude, center, width] All in MHz

        vl = y[2]*g # B*\gamma [MHZ]
        para_B = [0.0, y[1], vl, y[3]] # [offset, amplitude, center, width] All in MHz

        error += abs(yhat[0] - y[0])*(8.5-0.001) + sum(abs(_funcNoise(omega2,*para_B)-_funcNoise(omega2,*para_A)))*(omega2[1]-omega2[0])

    return error / len(yBatch)

class SLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        B_lims   = [520,536]
        W1_lims   = [0.004,0.009]

        self.g=1.0705e-3 # C-13 nuclear spin gyromagnetic ratio
        self.omega2 = 2*np.pi*torch.linspace(self.g*min(B_lims)-5*max(W1_lims), self.g*max(B_lims)+5*max(W1_lims),500).to(device)


    def _funcGauss(self, x,y0, a,xc,w):
        return y0+a*torch.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

    def _funcNoise(self, x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
        return y0 + self._funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

    def forward(self,yhatBatch,yBatch):
        # Y0, A, B, W1

        error = 0
        for yhat,y in zip(yhatBatch, yBatch):
            vl = yhat[2]*self.g # B*\gamma [MHZ]
            para_A = [0.0, yhat[1], vl, yhat[3]] # [offset, amplitude, center, width] All in MHz

            vl = y[2]*self.g # B*\gamma [MHZ]
            para_B = [0.0, y[1], vl, y[3]] # [offset, amplitude, center, width] All in MHz

            error += abs(yhat[0] - y[0])*(8.5-0.001) + sum(abs(self._funcNoise(self.omega2,*para_B)-self._funcNoise(self.omega2,*para_A)))*(self.omega2[1]-self.omega2[0])

        return error / len(yBatch)



class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # mp.set_start_method('forkserver')

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
        startTime = datetime.now()
    
        print("====================")
        print("RUN USING {} on device {}".format(sys.argv[1], device))
        print(startTime.strftime(timeFormat))
        print("====================")

        splits = ['train', 'valid', 'test']


        minMax = (  #from data creation
            (0.002, 0.008),
            (0.3, 0.7),
            (520, 536),
            (0.004, 0.009),
        )

        basePath = "data"


        fileDataset = np.load(os.path.join(basePath,conf.dataset))

        x = fileDataset['coh']
        y = fileDataset['par']

        y = y[:,0,:]    # equal for all nN

        allCollapseRanges = (
            (0, 101),
            (101, 152),
            (152, 193),
        )

        collapseRange = (
            min(allCollapseRanges[conf.rangeCollapse[0]] + allCollapseRanges[conf.rangeCollapse[1]-1]),
            max(allCollapseRanges[conf.rangeCollapse[0]] + allCollapseRanges[conf.rangeCollapse[1]-1]),
        )

        x = x[:, conf.rangeNN[0]:conf.rangeNN[1], collapseRange[0]:collapseRange[1]]

        x = x.reshape(x.shape[0], -1)

        if conf.normalizeY:
            for i in range(y.shape[1]):
                y[:,i] -= minMax[i][0]
                y[:,i] /= minMax[i][1] - minMax[i][0]
            # y[:,0] -= minMax[3][0]
            # y[:,0] /= minMax[3][1] - minMax[3][0]


        fileSplit = np.load(os.path.join(basePath,conf.split))

        x = {s: x[fileSplit[s]] for s in splits}
        y = {s: y[fileSplit[s]] for s in splits}



        doValid = False
        doSloss = False

        model = linearRegression(conf.dimX, conf.dimY).to(device)

        if conf.taskType == 'regressionL2':
            criterion = torch.nn.MSELoss()
        elif conf.taskType == 'regressionS':
            criterion = SLoss(device)

        if conf.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)
        elif conf.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)


        for epoch in range(conf.epochs):
            if conf.batchTraining:
                optimizer.zero_grad()

                outputs = model(torch.tensor(x['train'], requires_grad=True).to(device))
                targets = torch.tensor(y['train'], requires_grad=True).to(device)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
            else:
                batchX = torch.tensor(x['train'], requires_grad=True).to(device)
                batchY = torch.tensor(y['train'], requires_grad=True).to(device)
                for i in range(len(x['train'])):
                    optimizer.zero_grad()

                    outputs = model(batchX[i].reshape(1,-1))
                    targets = batchY[i].reshape(1,-1)
                    loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

            if doValid:
                with torch.no_grad(): # we don't need gradients in the testing phase
                    outputsVal = model(torch.tensor(x['valid'], requires_grad=True).to(device))
                    targetsVal = torch.tensor(y['valid'], requires_grad=True).to(device)
                    lossVal = criterion(outputsVal, targetsVal)


                print('epoch {}, train loss {:.2f}, val loss {:.2f}, train sloss {:.2f}, val sloss {:.2f}'.format(
                    epoch,
                    loss.item(),
                    lossVal.item(),
                    slossNP(outputs.cpu().data.numpy(), targets.cpu().data.numpy()),
                    slossNP(outputsVal.cpu().data.numpy(), targetsVal.cpu().data.numpy()),
                    # slossTorch(outputs, targets, device),
                    # slossTorch(outputsVal, targetsVal, device)
                ))
            else:
                if doSloss:
                    print('epoch {}, train loss {:.2f}, train sloss {:.2f}'.format(
                        epoch,
                        loss.item(),
                        slossNP(outputs.cpu().data.numpy(), targets.cpu().data.numpy())
                    ))
                else:
                    print('epoch {}, train loss {:.2f}'.format(
                        epoch,
                        loss.item()
                    ))


        with torch.no_grad(): # we don't need gradients in the testing phase
            predDict = {s: {
                'x': x[s],
                'y': y[s],
                'pred': model(torch.tensor(x[s], requires_grad=True).to(device)).cpu().data.numpy()
            } for s in splits}

        filesPath = os.path.join("files", conf.path)

        if not os.path.exists(os.path.join(filesPath, "predictions")):
            os.makedirs(os.path.join(filesPath, "predictions"))

        filePred = os.path.join(filesPath, "predictions", "linearRegressionPytorch.npz")

        np.savez_compressed(filePred, **predDict)
        print("Saved {}".format(filePred), flush=True)

        print("test sloss: {}".format(slossNP(predDict['test']['pred'], predDict['test']['y'])))

        endTime = datetime.now()
        print("=======")
        print("End {}".format(sys.argv[1]))
        print(endTime.strftime(timeFormat))
        print("====================")
        # notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),endTime.strftime(timeFormat),str(endTime-startTime).split('.', 2)[0]))
    
