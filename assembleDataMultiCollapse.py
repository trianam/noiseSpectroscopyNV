#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
import gzip
import os
from datetime import datetime
from itertools import product
import argparse


configurations = {
    'conf1': {
        'inputPath':    "simsMC",
        'outputPath':   "data",
        'filename64':   "dataset64MC.npz",
        'filename32':   "dataset32MC.npz",
        'noiseSD':        None,
    },
    'conf2': {
        'inputPath':    "simsMC",
        'outputPath':   "data",
        'filename64':   "dataset64MC-n025.npz",
        'filename32':   "dataset32MC-n025.npz",
        'noiseSD':        0.025,
    },
    'conf3': {
        'inputPath':    "simsMC",
        'outputPath':   "data",
        'filename64':   "dataset64MC-n01.npz",
        'filename32':   "dataset32MC-n01.npz",
        'noiseSD':        0.01,
    },
    'conf4': {
        'inputPath':    "simsMC",
        'outputPath':   "data",
        'filename64':   "dataset64MC-n05.npz",
        'filename32':   "dataset32MC-n05.npz",
        'noiseSD':        0.05,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assemble data multi collapse with given configuration.')
    parser.add_argument('conf', metavar='conf', type=str, nargs=1, help='The configuration key')
    args = parser.parse_args()

    conf = configurations[args.conf[0]]
    inputPath = conf['inputPath']
    outputPath = conf['outputPath']
    filename64 = conf['filename64']
    filename32 = conf['filename32']
    noiseSD = conf['noiseSD']

    os.makedirs(outputPath, exist_ok=True)

    valuesY0 = np.array([0.002, 0.00275, 0.0035, 0.00425, 0.005, 0.00575, 0.0065, 0.00725, 0.008]) #np.linspace(0.002, 0.008, 9),
    valuesA = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) #np.linspace(0.3, 0.7, 9),
    valuesB = np.array([520, 522, 524, 526, 528, 530, 532, 534, 536]) #np.linspace(520, 536, 9),
    valuesW1 = np.array([0.004, 0.004625, 0.00525, 0.005875, 0.0065, 0.007125, 0.00775, 0.008375, 0.009]) #np.linspace(0.004, 0.009, 9),
    valuesNN = np.array([1, 4, 8, 16, 24, 32, 40, 48, 56])

    numCombinations = len(valuesY0)*len(valuesA)*len(valuesB)*len(valuesW1)

    startTime = datetime.now()

    first = True
    for iNN, nN in enumerate(valuesNN):
        for iPar, par in enumerate(product(valuesY0, valuesA, valuesB, valuesW1)):
            parString = "{}_{}_{}_{}_{}".format(*par, nN)
            fileName = os.path.join(inputPath, "data_{}_.npy.gz".format(parString))
            coh = np.load(gzip.open(fileName, "rb"))
            par = np.array(par)
            print("{} ({}): Load {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), (iNN * numCombinations) + iPar + 1, fileName))
            if first:
                allCoh64 = np.zeros((numCombinations, len(valuesNN), coh.shape[0]), dtype=np.float64)
                allCoh32 = np.zeros((numCombinations, len(valuesNN), coh.shape[0]), dtype=np.float32)

                allPar64 = np.zeros((numCombinations, len(valuesNN), par.shape[0]), dtype=np.float64)
                allPar32 = np.zeros((numCombinations, len(valuesNN), par.shape[0]), dtype=np.float32)

                first = False

            allCoh64[iPar][iNN] = coh
            allCoh32[iPar][iNN] = coh

            allPar64[iPar][iNN] = par
            allPar32[iPar][iNN] = par

    if not noiseSD is None:
        noise = np.random.normal(0, noiseSD, allCoh32.shape)
        allCoh64 += noise
        allCoh32 += noise

    fileName = os.path.join(outputPath, filename64)
    print("Saving in {} ({} / {})".format(fileName, allCoh64.shape, allPar64.shape))
    np.savez_compressed(fileName, coh=allCoh64, par=allPar64)

    fileName = os.path.join(outputPath, filename32)
    print("Saving in {} ({} / {})".format(fileName, allCoh32.shape, allPar32.shape))
    np.savez_compressed(fileName, coh=allCoh32, par=allPar32)

    endTime = datetime.now()
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))