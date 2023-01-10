#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
import gzip
import os
from datetime import datetime
from multiprocessing import Pool
from itertools import product
import warnings

from calculate_chi_func import chiCalc_custom

def createSingleChi(processNum):
    rangeY0 = (0.0004, 0.004)
    rangeA = (0.3, 0.7)
    rangeB = (403, 403.4)
    rangeW1 = (0.002, 0.009)

    np.random.seed(processNum)
    y0 = np.random.uniform(rangeY0[0], rangeY0[1])
    a = np.random.uniform(rangeA[0], rangeA[1])
    B = np.random.uniform(rangeB[0], rangeB[1])
    w1 = np.random.uniform(rangeW1[0], rangeW1[1])

    exceptions = []
    # Parameters of the control field
    #t1_vec = np.arange(0.1, 1.5, 0.002)  # Se va troppo lento usa questo
    # t1_vec = np.arange(0.1,2,0.002) # time vector for t1 [µs]
    # center = 1/(4*0.0010705*B)
    # deltaT = 0.002
    # windowSize = 0.4

    valuesNN = np.array([1, 8, 16, 24, 32, 40, 48])
    t1_vec = np.array([1.65, 1.66, 1.67, 1.68, 1.69, 1.7 , 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8 , 1.81, 1.82, 1.83,
                       2.75, 2.76, 2.77, 2.78, 2.79, 2.8 , 2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87, 2.88, 2.89, 2.9 , 2.91, 2.92, 2.93, 2.94, 2.95, 2.96, 2.97, 2.98, 2.99, 3.  , 3.01, 3.02, 3.03, 3.04, 3.05])
    #concatenated windows for the 2 collapses: [1.65, 1.83], [2.75, 3.05]; indices: [0, 19), [19, 50)

    allChi = np.zeros((len(valuesNN), len(t1_vec)))
    for indexNN, nN in enumerate(valuesNN):
        totT_vec = 2 * nN * t1_vec  # total time vector [µs]

        # NSD function
        def funcGauss(x, y0, a, xc, w):
            return y0 + a * np.exp(
                -0.5 * ((x - 2 * np.pi * xc) / (2 * np.pi * w)) ** 2)  # I included a couple of 2*np.pi to convert \nu->


        def funcNoise(x, y0, a1, x1, w1):
            return y0 + funcGauss(x, 0, a1, x1,
                                  w1)  # + funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)


        ## Function to calculate the distribution of pi pulses for a CPMG sequence
        def cpmg(t1, nN):
            seq = np.ones(nN + 1) * 2 * t1
            seq[0] = t1
            seq[-1] = t1  # [-1] indica l'elemento finale del vettore
            return seq


        # showPlot = True  # False #
        # saveData = False  # True #

        # NSD parameters
        vl = B * 1.0705e-3  # B*\gamma [MHZ]
        para = np.array([y0, a, vl, w1])  # [offset, amplitude, center, width] all in MHz

        ## Calculate chi at every total time
        chi = np.zeros(len(totT_vec))

        for i, totT in enumerate(totT_vec):
            ## Time between pulses
            t1 = totT / (2 * nN)

            ## pi pulses distribution
            pulses_times = cpmg(t1, nN).cumsum()[:-1]  # [:-1] <--- del vettore considerato si prendono tutti i valori all'infuori dell'ultimo

            ## Calculate chi
            try:
                chi[i] = chiCalc_custom(funcNoise, para, totT, pulses_times)
            except Exception as e:
                exceptions.append((i, t1, e))

        allChi[indexNN] = np.exp(-chi)

    # parString = "{}_{}_{}_{}".format(y0, a, B, w1)

    # return np.float32(np.exp(-chi)), parString
    parameters = np.array([y0, a, B, w1])

    fileName = os.path.join(outputPath, "data_{}.npz".format(processNum))
    np.savez_compressed(fileName, coh=allChi, par=parameters)
    if len(exceptions)>0:
        os.makedirs(exceptionsPath, exist_ok=True)
        # exceptionsFileName = os.path.join(exceptionsPath, "exceptions_{}_{}_{}_{}.txt".format(*parameters))
        exceptionsFileName = os.path.join(exceptionsPath, "exceptions_{}.txt".format(processNum))
        with open(exceptionsFileName, "a+") as f:
            f.write('\n\n'.join(map(lambda e:"t1 {} ({}): {}".format(e[1],e[0],e[2]), exceptions)))
            f.write('\n')

    return fileName

if __name__ == "__main__":
    # warnings.filterwarnings('error')
    warnings.filterwarnings('ignore')

    workers = 70

    outputPath = "simsMCRnew"
    os.makedirs(outputPath, exist_ok=True)
    exceptionsPath = "exceptionsMCRnew"

    numCombinations = 100000

    #=============================================================================
    # createSingleChi(list(allCombinations)[1000])
    #=============================================================================
    startTime = datetime.now()

    startTimeIter = datetime.now()
    i = 1
    expTime = "0"
    pool = Pool(workers)
    for fileName in pool.imap_unordered(createSingleChi, range(numCombinations)):
        if (i-1) % 100 == 0:
            endTimeIter = datetime.now()
            expTime = str((endTimeIter-startTimeIter)/100 * (numCombinations - i)).split('.', 2)[0]
            startTimeIter = endTimeIter


        print("{} ({}) exp {} : Save {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i, expTime, fileName))
        i += 1

    endTime = datetime.now()
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))