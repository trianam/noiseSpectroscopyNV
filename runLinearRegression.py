#!/usr/bin/env python

import os
import sys
import socket
import configurations
import notifier
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

def _funcGauss(x,y0, a,xc,w):
    return y0+a*np.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

def _funcNoise(x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
    return y0 + _funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

def lossBAK(yhatBatch,yBatch):
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

def loss(conf, yhatBatch,yBatch):
    # Y0, A, B, W1

    B_lims   = [520,536]
    W1_lims   = [0.004,0.009]

    g=1.0705e-3 # C-13 nuclear spin gyromagnetic ratio
    omega2 = 2*np.pi*np.linspace(g*min(B_lims)-5*max(W1_lims), g*max(B_lims)+5*max(W1_lims),500)

    error = 0
    for yhat,y in zip(yhatBatch, yBatch):
        if conf.has("fixedB"):
            Y0 = y[0]
            Y0_hat = yhat[0]
            A = y[1]
            A_hat = yhat[1]
            B = conf.fixedB
            B_hat = conf.fixedB
            W1 = y[2]
            W1_hat = yhat[2]
        else:
            Y0 = y[0]
            Y0_hat = yhat[0]
            A = y[1]
            A_hat = yhat[1]
            B = y[2]
            B_hat = yhat[2]
            W1 = y[3]
            W1_hat = yhat[3]

        vl = B_hat*g # B*\gamma [MHZ]
        para_A = [0.0, A_hat, vl, W1_hat] # [offset, amplitude, center, width] All in MHz

        vl = B*g # B*\gamma [MHZ]
        para_B = [0.0, A, vl, W1] # [offset, amplitude, center, width] All in MHz

        error += abs(Y0_hat - Y0)*(8.5-0.001) + sum(abs(_funcNoise(omega2,*para_B)-_funcNoise(omega2,*para_A)))*(omega2[1]-omega2[0])

    return error / len(yBatch)


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
        confAll = eval('configurations.{}'.format(sys.argv[1]))
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

        values = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]])))
        # epsilons = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]]), 4))
        epsilons = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]]), 3)) #TODO: resolve this (3/4)

        valuesExp = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]])))
        # epsilonsExp = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]]), 4))
        epsilonsExp = np.zeros((len(confAll), len(confAll[list(confAll.keys())[0]]), 3)) #TODO: resolve this (3/4)


        # for n in reversed(range(len(confAll))):
        #     for c in reversed(range(len(confAll[n]))):
        # for i_n,n in enumerate(reversed(confAll)):
        #     for i_c,c in enumerate(reversed(confAll[n])):
        for i_n,n in enumerate(confAll):
            for i_c,c in enumerate(confAll[n]):
                print("Processing n: {} c: {}".format(n, c))
                conf = confAll[n][c]

                fileDataset = np.load(os.path.join(basePath,conf.dataset))

                x = fileDataset['coh']
                y = fileDataset['par']

                if len(y.shape) == 3:
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


                # reg = MultiOutputRegressor(LinearRegression()).fit(x['train'], y['train'])
                if conf.has("polyDegree"):
                    reg = Pipeline([('poly', PolynomialFeatures(degree=conf.polyDegree)), ('linear', LinearRegression(fit_intercept=False))])
                else:
                    reg = LinearRegression()

                reg.fit(x['train'], y['train'])
                pred = {s: reg.predict(x[s]) for s in splits}

                # loss = fun.getLoss(conf, device)
                # values[n][c] = loss(torch.tensor(pred['test']).to(device), torch.tensor(y['test']).to(device)).item()
                values[i_n][i_c] = loss(conf, pred['test'], y['test'])
                epsilons[i_n][i_c][:] = (y['test'] - pred['test']).mean(axis=0)
                
                if conf.has("datasetExperimental"):
                    fileDatasetExp = np.load(os.path.join(basePath,conf.datasetExperimental))

                    xExp = fileDatasetExp['coh']
                    yExp = fileDatasetExp['par']
    
                    if len(yExp.shape) == 3:
                        yExp = yExp[:,0,:]    # equal for all nN
    
                    xExp = xExp[:, conf.rangeNN[0]:conf.rangeNN[1], collapseRange[0]:collapseRange[1]]
    
                    xExp = xExp.reshape(xExp.shape[0], -1)
    
                    if conf.normalizeY:
                        for i in range(yExp.shape[1]):
                            yExp[:,i] -= minMax[i][0]
                            yExp[:,i] /= minMax[i][1] - minMax[i][0]

                    predExp = reg.predict(xExp)
    
                    valuesExp[i_n][i_c] = loss(conf, predExp, yExp)
                    epsilonsExp[i_n][i_c][:] = (yExp - predExp).mean(axis=0)

                    

        print(values)
        print(epsilons)

        if True:
            print("============================================ nkl")
            nns = [1,4,8,16,24,32,40,48,56]
            ks = [1,2,3]
            for i_n,n in enumerate(confAll):
                for i_c,c in enumerate(confAll[n]):
                    if len(confAll[n]) == 1:
                        print("{} {}".format(nns[n], values[i_n][i_c]))
                    else:
                        print("{} {} {}".format(nns[n], ks[c], values[i_n][i_c]))

                if len(confAll[n]) > 1:
                    print("")
            print("============================================ nkl experimental")
            nns = [1,4,8,16,24,32,40,48,56]
            ks = [1,2,3]
            for i_n,n in enumerate(confAll):
                for i_c,c in enumerate(confAll[n]):
                    if len(confAll[n]) == 1:
                        print("{} {}".format(nns[n], valuesExp[i_n][i_c]))
                    else:
                        print("{} {} {}".format(nns[n], ks[c], valuesExp[i_n][i_c]))

                if len(confAll[n]) > 1:
                    print("")
            print("============================================ Loss")
            nns = [1,4,8,16,24,32,40,48,56]
            for i_n,n in enumerate(confAll):
                print("$\\overline{{N}}={}$".format(nns[n]), end='')
                for i_c,c in enumerate(confAll[n]):
                    base,exp = np.format_float_scientific(values[i_n][i_c], 3, exp_digits=1).split('e')
                    print("& ${}\\times 10^{{{}}}$".format(base, exp), end='')
                print("\\\\")

            print("============================================ Eps mean")
            for i_n,n in enumerate(confAll):
                print("$\\overline{{N}}={}$".format(nns[n]), end='')
                for i_c,c in enumerate(confAll[n]):
                    base,exp = np.format_float_scientific(epsilons[i_n][i_c].mean(), 3, exp_digits=1).split('e')
                    print("& ${}\\times 10^{{{}}}$".format(base, exp), end='')
                print("\\\\")


            if confAll[list(confAll.keys())[0]][list(confAll[list(confAll.keys())[0]].keys())[0]].has("fixedB"):
                noiseParams = ['Y0', 'A', 'W1']
            else:
                noiseParams = ['Y0', 'A', 'B', 'W1']
            for parI, parName in enumerate(noiseParams):
                print("============================================ Eps {}".format(parName))
                for i_n,n in enumerate(confAll):
                    print("$\\overline{{N}}={}$".format(nns[n]), end='')
                    for i_c,c in enumerate(confAll[n]):
                        base,exp = np.format_float_scientific(epsilons[i_n][i_c][parI], 3, exp_digits=1).split('e')
                        print("& ${}\\times 10^{{{}}}$".format(base, exp), end='')
                    print("\\\\")

        endTime = datetime.now()
        print("=======")
        print("End {}".format(sys.argv[1]))
        print(endTime.strftime(timeFormat))
        print("====================")
        # notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),endTime.strftime(timeFormat),str(endTime-startTime).split('.', 2)[0]))
    
