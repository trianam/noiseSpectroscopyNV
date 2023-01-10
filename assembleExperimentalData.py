import os
import numpy as np
import re

experimentalPath = "experimentalData/experimental_data_no_nearby_C"
outputFile = "data/dataset32MCnoB-experimental.npz"

cohDict = {}
for f in os.listdir(experimentalPath):
    currArr = np.genfromtxt(os.path.join(experimentalPath, f), delimiter=" ")
    n = int(re.search('\d+', re.search('_n-\d+_', f).group(0)).group(0))
    cohDict[n] = currArr[1]

coh = np.zeros((1,9,193), dtype=np.float32)
par = np.zeros((1,9,3), dtype=np.float32)
for i,n in enumerate(sorted(cohDict)):
    print("{} -> {}".format(i,n))
    coh[0,i] = cohDict[n]
    par[0,i] = np.array([0.004, 0.48, 0.0062]) #y0, a, W1

np.savez(outputFile, coh=coh, par=par)
print("Saved in {}".format(outputFile))
