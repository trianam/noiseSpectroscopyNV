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

def createSingleChi(parameters): # nN: Number of pi pulses
    y0, a, B, w1, nN = parameters
    exceptions = []
    # Parameters of the control field
    #t1_vec = np.arange(0.1, 1.5, 0.002)  # Se va troppo lento usa questo
    # t1_vec = np.arange(0.1,2,0.002) # time vector for t1 [µs]
    center = 1/(4*0.0010705*B)
    deltaT = 0.002
    windowSize = 0.4
    t1_vec = np.array([0.2  , 0.204, 0.208, 0.212, 0.216, 0.22 , 0.224, 0.228, 0.232,
                   0.236, 0.24 , 0.244, 0.248, 0.252, 0.256, 0.26 , 0.264, 0.268,
                   0.272, 0.276, 0.28 , 0.284, 0.288, 0.292, 0.296, 0.3  , 0.304,
                   0.308, 0.312, 0.316, 0.32 , 0.324, 0.328, 0.332, 0.336, 0.34 ,
                   0.344, 0.348, 0.352, 0.356, 0.36 , 0.364, 0.368, 0.372, 0.376,
                   0.38 , 0.384, 0.388, 0.392, 0.396, 0.4  , 0.404, 0.408, 0.412,
                   0.416, 0.42 , 0.424, 0.428, 0.432, 0.436, 0.44 , 0.444, 0.448,
                   0.452, 0.456, 0.46 , 0.464, 0.468, 0.472, 0.476, 0.48 , 0.484,
                   0.488, 0.492, 0.496, 0.5  , 0.504, 0.508, 0.512, 0.516, 0.52 ,
                   0.524, 0.528, 0.532, 0.536, 0.54 , 0.544, 0.548, 0.552, 0.556,
                   0.56 , 0.564, 0.568, 0.572, 0.576, 0.58 , 0.584, 0.588, 0.592,
                   0.596, 0.6  , 1.1  , 1.108, 1.116, 1.124, 1.132, 1.14 , 1.148,
                   1.156, 1.164, 1.172, 1.18 , 1.188, 1.196, 1.204, 1.212, 1.22 ,
                   1.228, 1.236, 1.244, 1.252, 1.26 , 1.268, 1.276, 1.284, 1.292,
                   1.3  , 1.308, 1.316, 1.324, 1.332, 1.34 , 1.348, 1.356, 1.364,
                   1.372, 1.38 , 1.388, 1.396, 1.404, 1.412, 1.42 , 1.428, 1.436,
                   1.444, 1.452, 1.46 , 1.468, 1.476, 1.484, 1.492, 1.5  , 2.   ,
                   2.01 , 2.02 , 2.03 , 2.04 , 2.05 , 2.06 , 2.07 , 2.08 , 2.09 ,
                   2.1  , 2.11 , 2.12 , 2.13 , 2.14 , 2.15 , 2.16 , 2.17 , 2.18 ,
                   2.19 , 2.2  , 2.21 , 2.22 , 2.23 , 2.24 , 2.25 , 2.26 , 2.27 ,
                   2.28 , 2.29 , 2.3  , 2.31 , 2.32 , 2.33 , 2.34 , 2.35 , 2.36 ,
                   2.37 , 2.38 , 2.39 , 2.4  ])
    #concatenated windows for the 3 collapses: [0.2, 0.6], [1.1, 1.5], [2, 2.4]; indices: [0, 101), [101, 152), [152, 193)

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


    showPlot = True  # False #
    saveData = False  # True #

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


    parString = "{}_{}_{}_{}_{}".format(y0, a, B, w1, nN)

    # return np.float32(np.exp(-chi)), parString
    return np.exp(-chi), parString, exceptions

if __name__ == "__main__":
    # warnings.filterwarnings('error')

    workers = 70

    configurations = {
        'conf1': {
            'outputPath': "simsMC",
            'exceptionsPath': "exceptionsMC",
            'valuesY0': np.array([0.002, 0.00275, 0.0035, 0.00425, 0.005, 0.00575, 0.0065, 0.00725, 0.008]), #np.linspace(0.002, 0.008, 9),
            'valuesA': np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]), #np.linspace(0.3, 0.7, 9),
            'valuesB': np.array([520, 522, 524, 526, 528, 530, 532, 534, 536]), #np.linspace(520, 536, 9),
            'valuesW1': np.array([0.004, 0.004625, 0.00525, 0.005875, 0.0065, 0.007125, 0.00775, 0.008375, 0.009]), #np.linspace(0.004, 0.009, 9),
            'valuesNN': np.array([1, 4, 8, 16, 24, 32, 40, 48, 56]),
        },
        'conf11': {
            'outputPath': "simsMC1",
            'exceptionsPath': "exceptionsMC1",
            'valuesY0': np.array([0.002, 0.00275, 0.0035, 0.00425, 0.005, 0.00575, 0.0065, 0.00725, 0.008]), #np.linspace(0.002, 0.008, 9),
            'valuesA': np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]), #np.linspace(0.3, 0.7, 9),
            'valuesB': np.array([520, 522, 524, 526, 528, 530, 532, 534, 536]), #np.linspace(520, 536, 9),
            'valuesW1': np.array([0.004, 0.004625, 0.00525, 0.005875, 0.0065, 0.007125, 0.00775, 0.008375, 0.009]), #np.linspace(0.004, 0.009, 9),
            'valuesNN': np.array([1]),
        },
    }

    if len(sys.argv) != 2:
        raise Exception("Use {} configuration".format(sys.argv[0]))
    useConf = sys.argv[1]

    outputPath = configurations[useConf]['outputPath']
    os.makedirs(outputPath, exist_ok=True)
    exceptionsPath = configurations[useConf]['exceptionsPath']

    valuesY0 = configurations[useConf]['valuesY0']
    valuesA = configurations[useConf]['valuesA']
    valuesB = configurations[useConf]['valuesB']
    valuesW1 = configurations[useConf]['valuesW1']
    valuesNN = configurations[useConf]['valuesNN']

    allCombinations = product(valuesY0, valuesA, valuesB, valuesW1, valuesNN)
    numCombinations = len(valuesY0)*len(valuesA)*len(valuesB)*len(valuesW1)*len(valuesNN)

    #=============================================================================
    # createSingleChi(list(allCombinations)[1000])
    #=============================================================================
    startTime = datetime.now()

    startTimeIter = datetime.now()
    i = 1
    pool = Pool(workers)
    for coh,parString,exceptions in pool.imap_unordered(createSingleChi, allCombinations):
        if (i-1) % 100 == 0:
            endTimeIter = datetime.now()
            expTime = str((endTimeIter-startTimeIter)/100 * (numCombinations - i)).split('.', 2)[0]
            startTimeIter = endTimeIter
        # fileName = os.path.join(outputPath, "data_{}_.npy".format(parString))
        # np.save(open(fileName, 'wb'), coh)
        fileName = os.path.join(outputPath, "data_{}_.npy.gz".format(parString))
        np.save(gzip.GzipFile(fileName, "w"), coh)
        if len(exceptions)>0:
            os.makedirs(exceptionsPath, exist_ok=True)
            exceptionsFileName = os.path.join(exceptionsPath, "exceptions_{}.txt".format(parString))
            with open(exceptionsFileName, "a+") as f:
                f.write('\n\n'.join(map(lambda e:"t1 {} ({}): {}".format(e[1],e[0],e[2]), exceptions)))
                f.write('\n')

        print("{} ({}) exp {} : Save {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i, expTime, fileName))
        i += 1

    endTime = datetime.now()
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))