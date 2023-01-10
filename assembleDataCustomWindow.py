#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py
import gzip
import os
import sys
from datetime import datetime
from itertools import product


if __name__ == "__main__":
    outputPath = "data"

    confs = {
        'V3':{
            'inputPath': "sims",
            'numPoints': 40,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'indices': [
                [22, 30, 16, 32],
                [34, 17, 35, 3],
            ],
            'windows': [
                0.30+np.arange(201)*0.002,
                0.10+np.arange(201)*0.002,
            ]
        },
        'V4':{
            'inputPath': "sims20",
            'numPoints': 20,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
        },
    }

    if len(sys.argv) < 2 or not sys.argv[1] in confs:
        raise Exception("Use {} V? (with V? one of the available).".format(sys.argv[0]))

    useConf = sys.argv[1]

    inputPath = confs[useConf]['inputPath']
    numPoints = confs[useConf]['numPoints']
    numSamples = confs[useConf]['numSamples']
    typeSD = confs[useConf]['typeSD']

    np.random.seed(42)
    os.makedirs(outputPath, exist_ok=True)

    mins = np.array([0.0005, 0.1, 50, 0.001])
    maxs = np.array([0.01, 1, 1000, 0.01])

    valuesY0 = np.linspace(0.0005, 0.01, numPoints)
    valuesA = np.linspace(0.1, 1, numPoints)
    valuesB = np.linspace(50, 1000, numPoints)
    valuesW1 = np.linspace(0.001, 0.01, numPoints)
    # valuesNN = [8]

    allCombinations = [
        [ values[indices[i]] for i,values in enumerate((valuesY0,valuesA,valuesB,valuesW1)) ]
        for indices in confs[useConf]['indices']
    ]
    numCombinations = len(allCombinations)

    startTime = datetime.now()

    saveFileName = os.path.join(outputPath, "dataset32Custom{}.hdf5".format(useConf))
    print("Saving in {}".format(saveFileName))
    with h5py.File(saveFileName, 'w') as h5File:
        startTimeIter = startTime
        for i, par in enumerate(allCombinations):
            if i % 1000 == 0:
                endTimeIter = datetime.now()
                expTime = str((endTimeIter-startTimeIter)/1000 * (numCombinations-1 - i)).split('.', 2)[0]
                startTimeIter = endTimeIter
            parString = "{}_{}_{}_{}_8".format(*par)
            loadFileName = os.path.join(inputPath, "data_{}_.npy.gz".format(parString))
            print("{} ({}/{}; exp: {}): Load {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i+1, numCombinations, expTime, loadFileName))
            coh = np.load(gzip.open(loadFileName, "rb"))
            par = np.array(par)
            if i == 0:
                if splitted is None:
                    dataLen = numCombinations * numSamples
                    allCoh = h5File.create_dataset("coh", (dataLen, coh.shape[0]), dtype=np.float32)
                    allPar = h5File.create_dataset("par", (dataLen, par.shape[0]), dtype=np.float32)
                    allParN = h5File.create_dataset("parNorm", (dataLen, par.shape[0]), dtype=np.float32)

                    if not shuffleGroup is None:
                        if dataLen % shuffleGroup != 0:
                            raise ValueError("Data len {} and group {} are not compatible.".format(dataLen, shuffleGroup))
                        shuffleIndex = np.random.permutation(np.arange(dataLen).reshape(int(dataLen/shuffleGroup), shuffleGroup)).reshape(-1)
                        if not shuffleAfter is None:
                            testLen = int(dataLen / 100. * shuffleAfter)
                            np.random.shuffle(shuffleIndex[-testLen:])
                            np.random.shuffle(shuffleIndex[-2 * testLen: -testLen])
                            np.random.shuffle(shuffleIndex[: -2 * testLen])
                else:
                    with h5py.File(os.path.join(outputPath, splitted), 'r') as splitFile:
                        allCoh = {s: h5File.create_dataset("{}/coh".format(s), (len(splitFile[s]), coh.shape[0]), dtype=np.float32) for s in ['train', 'valid', 'test']}
                        allPar = {s: h5File.create_dataset("{}/par".format(s), (len(splitFile[s]), par.shape[0]), dtype=np.float32) for s in ['train', 'valid', 'test']}
                        allParN = {s: h5File.create_dataset("{}/parNorm".format(s), (len(splitFile[s]), par.shape[0]), dtype=np.float32) for s in ['train', 'valid', 'test']}

                        splitArray = {s: splitFile[s][:] for s in splitFile}
                        startTimeIterIndex = datetime.now()
                        numTotalValues = sum([len(splitFile[s]) for s in splitFile])
                        indexSet = np.zeros((numTotalValues,), dtype=str)
                        indexNum = np.zeros((numTotalValues,), dtype=int)
                        for j in range(numTotalValues):
                            if j % 100 == 0:
                                endTimeIterIndex = datetime.now()
                                expTime = str((endTimeIterIndex-startTimeIterIndex)/100 * (numTotalValues-1 - j)).split('.', 2)[0]
                                startTimeIterIndex = endTimeIterIndex
                                print("Building index {}/{}, exp: {}                             ".format(j, numTotalValues, expTime), end='\r')
                            for s in splitFile:
                                w = np.where(splitArray[s] == j)[0]
                                if w.shape[0] == 1:
                                    # index.append((s,w[0]))
                                    indexSet[j] = s
                                    indexNum[j] = w[0]
                                    break

            for s in range(numSamples):
                if splitted is None:
                    writeIndex = (i*numSamples)+s
                    if not shuffleGroup is None:
                        writeIndex = shuffleIndex[writeIndex]

                    if typeSD == 'single':
                        allCoh[writeIndex] = np.random.default_rng().normal(coh, confs[useConf]['samplesSD'])
                    elif typeSD == 'sampled':
                        allCoh[writeIndex] = np.random.default_rng().normal(coh, np.random.default_rng().normal(confs[useConf]['samplesSDm'], confs[useConf]['samplesSDstd'], len(coh)))
                    else:
                        raise ValueError("typeSD {} not valid".format(typeSD))
                    allPar[writeIndex] = par
                    allParN[writeIndex] = (par - mins) / (maxs - mins)
                    # for k in range(par.shape[0]):
                    #     allParN[(i*numSamples)+s, k] = (par[k] - minMax[k][0]) / (minMax[k][1] - minMax[k][0])
                else:
                    # convSet,convIndex = index[(i*numSamples)+s]
                    convSet = indexSet[(i*numSamples)+s]
                    convIndex = indexNum[(i*numSamples)+s]

                    if typeSD == 'single':
                        allCoh[convSet][convIndex] = np.random.default_rng().normal(coh, confs[useConf]['samplesSD'])
                    elif typeSD == 'sampled':
                        allCoh[convSet][convIndex] = np.random.default_rng().normal(coh, np.random.default_rng().normal(confs[useConf]['samplesSDm'], confs[useConf]['samplesSDstd'], len(coh)))
                    else:
                        raise ValueError("typeSD {} not valid".format(typeSD))
                    allPar[convSet][convIndex] = par
                    allParN[convSet][convIndex] = (par - mins) / (maxs - mins)


    endTime = datetime.now()
    print("Saved in {}".format(saveFileName))
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))