#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gzip
import os
from datetime import datetime
from multiprocessing import Pool
from itertools import product

from calculate_chi_func import chiCalc_custom

def createSingleChi(parameters): # nN: Number of pi pulses
    y0, a, B, w1, nN = parameters
    # Parameters of the control field
    #t1_vec = np.arange(0.1, 1.5, 0.002)  # Se va troppo lento usa questo
    # t1_vec = np.arange(0.1,2,0.002) # time vector for t1 [µs]
    center = 1/(4*0.0010705*B)
    deltaT = 0.002
    windowSize = 0.4
    t1_vec = np.arange(0, windowSize+deltaT, deltaT) + center - windowSize/2 # time vector for t1 [µs]
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
        chi[i] = chiCalc_custom(funcNoise, para, totT, pulses_times)

    parString = "{}_{}_{}_{}_{}".format(y0, a, B, w1, nN)

    # return np.float32(np.exp(-chi)), parString
    return np.exp(-chi), parString

if __name__ == "__main__":
    # numSamples = 2
    workers = 70

    outputPath = "simsWin"

    os.makedirs(outputPath, exist_ok=True)

    valuesY0 = np.linspace(0.0005, 0.01, 40)
    valuesA = np.linspace(0.1, 1, 10)
    # valuesB = np.linspace(50, 1000, numSamples)
    valuesB = np.linspace(120, 1000, 40)
    valuesW1 = np.linspace(0.001, 0.01, 10)
    valuesNN = [8]

    allCombinations = product(valuesY0, valuesA, valuesB, valuesW1, valuesNN)
    numCombinations = len(valuesY0)*len(valuesA)*len(valuesB)*len(valuesW1)#*len(valuesNN)

    startTime = datetime.now()

    startTimeIter = datetime.now()
    i = 1
    pool = Pool(workers)
    for coh,parString in pool.imap_unordered(createSingleChi, allCombinations):
        if (i-1) % 100 == 0:
            endTimeIter = datetime.now()
            expTime = str((endTimeIter-startTimeIter)/100 * (numCombinations - i)).split('.', 2)[0]
            startTimeIter = endTimeIter
        # fileName = os.path.join(outputPath, "data_{}_.npy".format(parString))
        # np.save(open(fileName, 'wb'), coh)
        fileName = os.path.join(outputPath, "data_{}_.npy.gz".format(parString))
        np.save(gzip.GzipFile(fileName, "w"), coh)
        print("{} ({}) exp {} : Save {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i, expTime, fileName))
        i += 1

    endTime = datetime.now()
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))