#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys

confs = {
    "R1": {
        'noiseParameters': { # y_0, a, B, w_1
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
        },
        'window': None,
        'meanSample' :False,
        'numExtraSamples' :10,
    },
    "R1b": {
        'noiseParameters': { # y_0, a, B, w_1
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
        },
        'window': None,
        'meanSample' :True,
        'numExtraSamples' :0,
    },
    "R2.1": {
        'noiseParameters': {
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            # "XY8_N8_404G_newNV.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_404G_newNV.dat": [0,302],
        },
        'meanSample' :False,
        'numExtraSamples' :10,
    },
    "R2.2": {
        'noiseParameters': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
        },
        'window': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0,201],
        },
        'meanSample' :False,
        'numExtraSamples' :10,
    },
    "R2.3": {
        'noiseParameters': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
            # "XY8_N8_632G_from-old-data_interpolated.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [0,346],
        },
        'meanSample' :False,
        'numExtraSamples' :10,
    },
    "R3.1": {
        'noiseParameters': {
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            # "XY8_N8_404G_newNV.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_404G_newNV.dat": [0,201],
        },
        'meanSample' :True,
        'numExtraSamples' :10,
    },
    "R3.2": {
        'noiseParameters': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
        },
        'window': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0,201],
        },
        'meanSample' :True,
        'numExtraSamples' :10,
    },
    "R3.3": {
        'noiseParameters': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
            # "XY8_N8_632G_from-old-data_interpolated.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [15, 216],
        },
        'meanSample' :True,
        'numExtraSamples' :10,
    },
    "R4.1": {
        'noiseParameters': {
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            # "XY8_N8_404G_newNV.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_404G_newNV.dat": [0,201],
        },
        'meanSample' :False,
        'numExtraSamples' :32,
    },
    "R4.2": {
        'noiseParameters': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
        },
        'window': {
            "XY8_N8_528G_from-old-data_interpolated.dat": [0,201],
        },
        'meanSample' :False,
        'numExtraSamples' :32,
    },
    "R4.3": {
        'noiseParameters': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
            # "XY8_N8_632G_from-old-data_interpolated.dat": [0,0,528,0],
        },
        'window': {
            "XY8_N8_632G_from-old-data_interpolated.dat": [15, 216],
        },
        'meanSample' :False,
        'numExtraSamples' :32,
    },
    "Ra": {
        'noiseParameters': {
            "XY8_N8_404G_newNV.dat": [0.00119,0.52,404,0.0042],
            "XY8_N8_528G_from-old-data_interpolated.dat": [0.004,0.48,528,0.0062],
            "XY8_N8_632G_from-old-data_interpolated.dat": [0.0037,0.38,632,0.0085],
        },
        'window': {
            "XY8_N8_404G_newNV.dat": [0,201],
            "XY8_N8_528G_from-old-data_interpolated.dat": [0,201],
            "XY8_N8_632G_from-old-data_interpolated.dat": [15, 216],
        },
        'meanSample' :True,
        'numExtraSamples' :0,
    },
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1] in confs:
        raise Exception("Use {} V? (with V? one of the available).".format(sys.argv[0]))
    useConf = sys.argv[1]

    noiseParameters = confs[useConf]['noiseParameters']
    window = confs[useConf]['window']
    meanSample = confs[useConf]['meanSample']
    numExtraSamples = confs[useConf]['numExtraSamples']

    inputPath = "real"
    outputPath = "data"
    outputFileName = "dataset32-REAL-{}.npz".format(useConf)

    np.random.seed(42)

    t1_vec = np.arange(0.1,2,0.002)
    eps = 10**-12

    os.makedirs(outputPath, exist_ok=True)

    x = []
    y = []

    for fileName in noiseParameters:
        t1,coh,coh_err = np.loadtxt(os.path.join(inputPath, fileName))

        if meanSample:
            if window is None:
                currX = np.zeros(len(t1_vec))
                for t,c in zip(t1,coh):
                    pos = np.argwhere((t1_vec > t-eps) & (t1_vec < t+eps))
                    if pos.shape != (1,1):
                        raise Exception("Something is wrong")
                    currX[pos[0,0]] = c
                x.append(currX)
                y.append(noiseParameters[fileName])
            else:
                minIndex, maxIndex = window[fileName]
                x.append(coh[minIndex:maxIndex])
                y.append(noiseParameters[fileName])

        for _ in range(numExtraSamples):
            if window is None:
                currX = np.zeros(len(t1_vec))
                for t,c,e in zip(t1,coh,coh_err):
                    pos = np.argwhere((t1_vec > t-eps) & (t1_vec < t+eps))
                    if pos.shape != (1,1):
                        raise Exception("Something is wrong")
                    currX[pos[0,0]] = np.random.default_rng().normal(c, e)
                x.append(currX)
                y.append(noiseParameters[fileName])
            else:
                minIndex, maxIndex = window[fileName]
                currX = np.zeros(maxIndex-minIndex)
                for t,(c,e) in enumerate(zip(coh[minIndex:maxIndex],coh_err[minIndex:maxIndex])):
                    currX[t] = np.random.default_rng().normal(c, e)

                x.append(currX)
                y.append(noiseParameters[fileName])


    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    np.savez_compressed(os.path.join(outputPath, outputFileName), coh=x, par=y)
    print("Saved in {}".format(os.path.join(outputPath, outputFileName)))
