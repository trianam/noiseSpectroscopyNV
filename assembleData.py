#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gzip
import os
from datetime import datetime
from itertools import product


if __name__ == "__main__":
    numSamples = 40
    inputPath = "sims"
    outputPath = "data"

    os.makedirs(outputPath, exist_ok=True)

    valuesY0 = np.linspace(0.0005, 0.01, numSamples)
    valuesA = np.linspace(0.1, 1, numSamples)
    valuesB = np.linspace(50, 1000, numSamples)
    valuesW1 = np.linspace(0.001, 0.01, numSamples)
    valuesNN = [8]

    allCombinations = product(valuesY0, valuesA, valuesB, valuesW1, valuesNN)
    numCombinations = len(valuesY0)*len(valuesA)*len(valuesB)*len(valuesW1)*len(valuesNN)

    startTime = datetime.now()

    for i, par in enumerate(allCombinations):
        parString = "{}_{}_{}_{}_{}".format(*par)
        fileName = os.path.join(inputPath, "data_{}_.npy.gz".format(parString))
        coh = np.load(gzip.open(fileName, "rb"))
        par = np.array(par)
        print("{} ({}): Load {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i+1, fileName))
        if i == 0:
            allCoh64 = np.zeros((numCombinations, coh.shape[0]), dtype=np.float64)
            allCoh32 = np.zeros((numCombinations, coh.shape[0]), dtype=np.float32)

            allPar64 = np.zeros((numCombinations, par.shape[0]), dtype=np.float64)
            allPar32 = np.zeros((numCombinations, par.shape[0]), dtype=np.float32)

        allCoh64[i] = coh
        allCoh32[i] = coh

        allPar64[i] = par
        allPar32[i] = par


    fileName = os.path.join(outputPath, "dataset64.npz")
    print("Saving in {}".format(fileName))
    np.savez_compressed(fileName, coh=allCoh64, par=allPar64)

    fileName = os.path.join(outputPath, "dataset32.npz")
    print("Saving in {}".format(fileName))
    np.savez_compressed(fileName, coh=allCoh32, par=allPar32)

    endTime = datetime.now()
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))