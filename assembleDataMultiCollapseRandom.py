#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gzip
import os
from datetime import datetime
from itertools import product
import argparse



configurations = {
    'conf1': {
        'inputPath':    "simsMCR",
        'outputPath':   "data",
        'filename64':   "dataset64MCR.npz",
        'filename32':   "dataset32MCR.npz",
        'noiseSD':      None,
        'numCombinations': 100000,
    },
    'conf2': {
        'inputPath':    "simsMCR",
        'outputPath':   "data",
        'filename64':   "dataset64MCR-n.npz",
        'filename32':   "dataset32MCR-n.npz",
        'noiseSD':      0.025,
        'numCombinations': 100000,
    },
    'confNew1': {
        'inputPath':    "simsMCRnew",
        'outputPath':   "data",
        'filename64':   "dataset64MCRnew.npz",
        'filename32':   "dataset32MCRnew.npz",
        'noiseSD':      0.05,
        'numCombinations': 100000,
    },
    'confNew2': {
        'inputPath':    "simsMCRnew",
        'outputPath':   "data",
        'filename64':   "dataset64MCRnew1000.npz",
        'filename32':   "dataset32MCRnew1000.npz",
        'noiseSD':      0.05,
        'numCombinations': 1000,
    },
    'confNew3': {
        'inputPath':    "simsMCRnew",
        'outputPath':   "data",
        'filename64':   "dataset64MCRnew10000.npz",
        'filename32':   "dataset32MCRnew10000.npz",
        'noiseSD':      0.05,
        'numCombinations': 10000,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assemble data multi collapse random generated with given configuration.')
    parser.add_argument('conf', metavar='conf', type=str, nargs=1, help='The configuration key')
    args = parser.parse_args()

    conf = configurations[args.conf[0]]
    inputPath = conf['inputPath']
    outputPath = conf['outputPath']
    filename64 = conf['filename64']
    filename32 = conf['filename32']
    noiseSD = conf['noiseSD']
    numCombinations = conf['numCombinations']

    os.makedirs(outputPath, exist_ok=True)

    startTime = datetime.now()

    first = True
    for i in range(numCombinations):
        fileName = os.path.join(inputPath, "data_{}.npz".format(i))
        current = np.load(fileName)
        coh = current['coh']
        par = current['par']
        print("{} ({}): Load {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i, fileName))
        if first:
            allCoh64 = np.zeros((numCombinations, coh.shape[0], coh.shape[1]), dtype=np.float64)
            allCoh32 = np.zeros((numCombinations, coh.shape[0], coh.shape[1]), dtype=np.float32)

            allPar64 = np.zeros((numCombinations, par.shape[0]), dtype=np.float64)
            allPar32 = np.zeros((numCombinations, par.shape[0]), dtype=np.float32)

            first = False

        allCoh64[i] = coh
        allCoh32[i] = coh

        allPar64[i] = par
        allPar32[i] = par

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