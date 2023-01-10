import os
import numpy as np
import re

experimentalPath = "experimentalData/experimental_data_2021"
outputFile = "data/dataset32MCnoB-experimental2021.npz"

cohDict = {}
for f in os.listdir(experimentalPath):
    currArr = np.genfromtxt(os.path.join(experimentalPath, f), delimiter=" ")
    n = int(re.search('\d+', re.search('_N\d+\.', f).group(0)).group(0))
    cohDict[n] = currArr[1]

coh = np.zeros((1,7,50), dtype=np.float32)
par = np.zeros((1,7,4), dtype=np.float32)
for i,n in enumerate(sorted(cohDict)):
    print("{} -> {}".format(i,n))
    coh[0,i] = cohDict[n]
    par[0,i] = np.array([0.00119, 0.52, 403.7, 0.0042]) #y0, a, B, W1

np.savez(outputFile, coh=coh, par=par)
print("Saved in {}".format(outputFile))
