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
        'V1':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 40),
            'valuesB': np.linspace(50, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 40),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'single',
            'samplesSD': 0.2,
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': False,
        },
        'V2':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 40),
            'valuesB': np.linspace(50, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 40),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': False,
        },
        'V3':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 40),
            'valuesB': np.linspace(50, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 40),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': 10,
            'shuffleAfter': 20,
            'customWindow': False,
        },
        'V3valid':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 40),
            'valuesB': np.linspace(50, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 40),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': True,
            'indices': [
                [22, 30, 16, 32],
                [34, 17, 35, 3],
            ],
            'windows': [
                [100, 300],
                [0, 200],
            ],
        },
        'V4':{
            'inputPath': "sims20",
            'valuesY0': np.linspace(0.0005, 0.01, 20),
            'valuesA': np.linspace(0.1, 1, 20),
            'valuesB': np.linspace(50, 1000, 20),
            'valuesW1': np.linspace(0.001, 0.01, 20),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': 10,
            'shuffleAfter': 20,
            'customWindow': False,
        },
        'V2test':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 2),
            'valuesA': np.linspace(0.1, 1, 2),
            'valuesB': np.linspace(50, 1000, 2),
            'valuesW1': np.linspace(0.001, 0.01, 2),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': False,
        },
        'V3test':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 2),
            'valuesA': np.linspace(0.1, 1, 2),
            'valuesB': np.linspace(50, 1000, 2),
            'valuesW1': np.linspace(0.001, 0.01, 2),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'single',
            'samplesSD': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': False,
        },
        'V4test':{
            'inputPath': "sims",
            'valuesY0': np.linspace(0.0005, 0.01, 2),
            'valuesA': np.linspace(0.1, 1, 2),
            'valuesB': np.linspace(50, 1000, 2),
            'valuesW1': np.linspace(0.001, 0.01, 2),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'single',
            'samplesSD': 0.,      # calculated from real/XY8_N8_404G_newNV.dat
            'splitted': None,
            'shuffleGroup': None,
            'shuffleAfter': None,
            'customWindow': False,
        },
        'W1':{ #windows
            'inputPath': "simsWin",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 10),
            'valuesB': np.linspace(120, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 10),
            'excludeThreshold': None,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': 10,
            'shuffleAfter': 20,
            'customWindow': False,
        },
        'W2':{ # windows filtered outside 0.5
            'inputPath': "simsWin",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 10),
            'valuesB': np.linspace(120, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 10),
            'excludeThreshold': 0.5,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': 10,
            'shuffleAfter': 20,
            'customWindow': False,
        },
        'W3':{ # windows filtered outside 0.4
            'inputPath': "simsWin",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 10),
            'valuesB': np.linspace(120, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 10),
            'excludeThreshold': 0.4,
            'numSamples': 10,
            'typeSD': 'sampled',
            'samplesSDm': 0.06741418153655117,      # calculated from real/XY8_N8_404G_newNV.dat
            'samplesSDstd': 0.0021935302815834965,
            'splitted': None,
            'shuffleGroup': 10,
            'shuffleAfter': 20,
            'customWindow': False,
        },
        'W4':{ # windows without sampling, only average
            'inputPath': "simsWin",
            'valuesY0': np.linspace(0.0005, 0.01, 40),
            'valuesA': np.linspace(0.1, 1, 10),
            'valuesB': np.linspace(120, 1000, 40),
            'valuesW1': np.linspace(0.001, 0.01, 10),
            'excludeThreshold': None,
            'numSamples': 1,
            'typeSD': 'none',
            'splitted': None,
            'shuffleGroup': 1,
            'shuffleAfter': None,
            'customWindow': False,
        },
    }

    if len(sys.argv) < 2 or not sys.argv[1] in confs:
        raise Exception("Use {} V? (with V? one of the available).".format(sys.argv[0]))

    useConf = sys.argv[1]

    inputPath = confs[useConf]['inputPath']
    numSamples = confs[useConf]['numSamples']
    typeSD = confs[useConf]['typeSD']
    splitted = confs[useConf]['splitted']
    shuffleGroup = confs[useConf]['shuffleGroup']
    shuffleAfter = confs[useConf]['shuffleAfter']
    customWindow = confs[useConf]['customWindow']

    valuesY0 = confs[useConf]['valuesY0']
    valuesA = confs[useConf]['valuesA']
    valuesB = confs[useConf]['valuesB']
    valuesW1 = confs[useConf]['valuesW1']
    # valuesNN = [8]

    excludeThreshold = confs[useConf]['excludeThreshold']

    np.random.seed(42)
    os.makedirs(outputPath, exist_ok=True)

    mins = np.array([min(values) for values in [valuesY0, valuesA, valuesB, valuesW1] ])
    maxs = np.array([max(values) for values in [valuesY0, valuesA, valuesB, valuesW1] ])



    if customWindow:
        allCombinations = [
            [ values[indices[i]] for i,values in enumerate((valuesY0,valuesA,valuesB,valuesW1)) ]
            for indices in confs[useConf]['indices']
        ]
        numCombinations = len(allCombinations)
        windows = confs[useConf]['windows']
    else:
        allCombinations = product(valuesY0, valuesA, valuesB, valuesW1)#, valuesNN)
        numCombinations = len(valuesY0)*len(valuesA)*len(valuesB)*len(valuesW1)#*len(valuesNN)

    if not excludeThreshold is None:
        filteredCombinations = []
        for c in allCombinations:
            minA = mins[1]
            maxA = maxs[1]
            minW1 = mins[3]
            maxW1 = maxs[3]

            A = (c[1] - minA) / (maxA - minA)
            W1 = (c[3] - minW1) / (maxW1 - minW1)
            if W1 < A + excludeThreshold + 0.01 and W1 > A - excludeThreshold - 0.01:
                filteredCombinations.append(c)

        allCombinations = filteredCombinations
        numCombinations = len(allCombinations)

    startTime = datetime.now()

    saveFileName = os.path.join(outputPath, "dataset32Aug{}.hdf5".format(useConf))
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
                        allCoh[writeIndex] = np.random.default_rng().normal(
                            coh, np.random.default_rng().normal(
                                confs[useConf]['samplesSDm'], confs[useConf]['samplesSDstd'], len(coh)
                            )
                        )
                    elif typeSD == 'none':
                        allCoh[writeIndex] = coh
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
                    elif typeSD == 'none':
                        allCoh[writeIndex] = coh
                    else:
                        raise ValueError("typeSD {} not valid".format(typeSD))
                    allPar[convSet][convIndex] = par
                    allParN[convSet][convIndex] = (par - mins) / (maxs - mins)


    endTime = datetime.now()
    print("Saved in {}".format(saveFileName))
    print("{}: End, started {} ({})".format(endTime.strftime("%d/%m/%Y %H:%M:%S"), startTime.strftime("%d/%m/%Y %H:%M:%S"), str(endTime-startTime).split('.', 2)[0]))