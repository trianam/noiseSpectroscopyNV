#     tuneConfigurations.py
#     The configurations for all the experiments.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from conf import Conf
from ray import tune
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
from collections import defaultdict

multiCollapseMinMax1 = (  #from data creation
        (0.002, 0.008),
        (0.3, 0.7),
        (520, 536),
        (0.004, 0.009),
    )

multiCollapseRanges1 = (
        (0, 101),
        (101, 152),
        (152, 193),
    )

# for multiCollapseRandomNew
multiCollapseMinMax2 = (  #from data creation
        (0.0004, 0.004),    #Y0
        (0.3, 0.7),         # A
        (403, 403.4),       # B
        (0.002, 0.009),     #W1
    )

multiCollapseRanges2 = (
        (0, 19),
        (19, 50),
    )


configGlobal = Conf({
    "debug":                False,
    "force":                False,
    "taskType":             "regressionL2", #classification, regressionL1, regressionL2, regressionMRE, regressionS
    "datasetType":          "classic",  # classic, multiCollapse
    "nonVerbose":           False,
    "useTune":              False,
    "tuneHyperOpt":         True,
    "tuneNumSamples":       1000, #if tuneHyperOpt then it is total number, else it is number for each point in grid
    "tuneParallelProc":     10,
    "tuneGrace":            2,
    "tuneGpuNum":           0,
    "optimizer":            'adam', #sgdtuneGpuNum or adam
    "learningRate":         0.001,
    "inputB":               False,
    "dimX":                 950,
    "dimY":                 4,
    "normalizeY":           True,
    "model":                'modelMLP',
    "activation":           'relu',
    "finalActivation":      'sigmoid',
    "hiddenLayers":         2,
    "hiddenDim":            950,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            16,
    "startEpoch":           0,
    "epochs":               100,
    "trackMetric":          "mse",
    "earlyStopping":        None,                          #None or patience
    "tensorBoard":          True,
    "logCurves":            True,
    "logEveryBatch":        False,
    "modelSave":            "best",
    "bestKey":              "mse",
    "bestSign":             "<",
    "modelLoad":            'last.pt',
    "shuffleDataset":       True,
    "filePredAppendix":     None,
    "customValidTest":      None,
})


# ================================================
config1 = configGlobal.copy({
    "path":                 'config1',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
})
config1continue = config1.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
config1load = config1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

config2 = configGlobal.copy({
    "path":                 'config2',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "hiddenLayers":         1,
    "epochs":               500,
})
config2continue = config2.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
config2load = config2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
config2loadReal = config2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL.npz',
    "split":                None,
    "shuffleDataset":       False,
})

# ================================================
configTest1 = configGlobal.copy({
    "force":                True,
    "path":                 'configTest1',
    "dataset":              'dataset32-TEST.npz',
    "split":                'dataset32-TEST-split.npz',
})
configTest1continue = configTest1.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configTest1load = configTest1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})


configTest2 = configGlobal.copy({
    "path":                 'configTest2',
    "dataset":              'dataset32-TEST2.npz',
    "split":                'dataset32-TEST2-split.npz',
})




# ================================================
configMRE1 = configGlobal.copy({
    "path":                 'configMRE1',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "epochs":               500,
    "hiddenLayers":         1,
    "taskType":             "regressionMRE",
    "finalActivation":      'none',
    "normalizeY":           False,
    "bestKey":              "mre",
    "trackMetric":          "mre",
})
configMRE1continue = configMRE1.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configMRE1load = configMRE1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})


configMRE2 = configGlobal.copy({
    "path":                 'configMRE2',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "epochs":               500,
    "hiddenLayers":         2,
    "taskType":             "regressionMRE",
    "finalActivation":      'none',
    "normalizeY":           False,
    "bestKey":              "mre",
})
configMRE2continue = configMRE2.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configMRE2load = configMRE2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configTestMRE1 = configGlobal.copy({
    "force":                True,
    "path":                 'configTestMRE1',
    "dataset":              'dataset32-TEST.npz',
    "split":                'dataset32-TEST-split.npz',
    "hiddenLayers":         2,
    "taskType":             "regressionMRE",
    "finalActivation":      'none',
    # "finalActivation":      'relu',
    "normalizeY":           False,
    "bestKey":              "mre",
})

# ================================================
configCNN1 = configGlobal.copy({
    "path":                 'configCNN1',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "model":                'modelCNN',
    "filters":              24,
    "pooling":              'max',

})
configCNN1continue = configCNN1.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configCNN1load = configCNN1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configCNN1loadReal = configCNN1.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL.npz',
    "split":                None,
    "shuffleDataset":       False,
})

configTestCNN1 = configCNN1.copy({
    "force":                True,
    "path":                 'configTestCNN1',
    "dataset":              'dataset32-TEST.npz',
    "split":                'dataset32-TEST-split.npz',
})

#FAKE
configCNN1b = configGlobal.copy({
    "path":                 'configCNN1b',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "model":                'modelCNN',
    "filters":              24,
    "pooling":              'max',

})

#--------------------------------------------------------------
configCNNmre1 = configGlobal.copy({
    "path":                 'configCNNmre1',
    "dataset":              'dataset32.npz',
    "split":                'dataset32-split.npz',
    "model":                'modelCNN',
    "filters":              24,
    "pooling":              'max',
    "taskType":             "regressionMRE",
    "finalActivation":      'none',
    "normalizeY":           False,
    "bestKey":              "mre",
    "trackMetric":          "mre",    
})
configCNNmre1continue = configCNNmre1.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configCNNmre1load = configCNNmre1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configCNNmre1loadReal = configCNNmre1.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL.npz',
    "split":                None,
    "shuffleDataset":       False,
})

configTestCNNmre1 = configCNNmre1.copy({
    "force":                True,
    "path":                 'configTestCNNmre1',
    "dataset":              'dataset32-TEST.npz',
    "split":                'dataset32-TEST-split.npz',
})

# ================================================
configCNN2 = configGlobal.copy({
    "path":                 'configCNN2',
    "dataset":              'dataset32AugV2.hdf5',
    "split":                'dataset32AugV2-split.hdf5',
    "model":                'modelCNN2',
    "dropout":              0.5,
})
configCNN2continue = configCNN2.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configCNN2load = configCNN2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configCNN2loadReal = configCNN2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL.npz',
    "split":                None,
    "shuffleDataset":       False,
})

# ================================================
configCNN3 = configGlobal.copy({
    "path":                 'configCNN3',
    "dataset":              'dataset32AugV2.hdf5',
    "split":                'dataset32AugV2-split.hdf5',
    "model":                'modelCNN',
    "filters":              24,
    "dropout":              0.5,
    "pooling":              'avg',
})
configCNN3continue = configCNN3.copy({
    "modelLoad":            'last.pt',
    "startEpoch":           -1,
})
configCNN3load = configCNN3.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configCNN3loadReal = configCNN3.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL.npz',
    "split":                None,
    "shuffleDataset":       False,
})

# ================================================
configCNN4 = configGlobal.copy({
    "path":                 'configCNN4',
    "dataset":              'dataset32AugV2.hdf5',
    "split":                'dataset32AugV2-split.hdf5',
    "batchSize":            256,
    "model":                'modelCNN',
    "filters":              40,
    "dropout":              0.5,
    "pooling":              'avg',
})

# ================================================
configCNN5 = configGlobal.copy({
    "path":                 'configCNN5',
    "dataset":              'dataset32AugV4.hdf5',
    "split":                20,
    "batchSize":            256,
    "model":                'modelCNN',
    "filters":              40,
    "dropout":              0.5,
    "pooling":              'avg',
})
configCNN5load = configCNN5.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configCNN6 = configGlobal.copy({
    "path":                 'configCNN6',
    "dataset":              'dataset32AugV4.hdf5',
    "split":                20,
    "batchSize":            256,
    "model":                'modelCNN3',
    "filters":              16,
    "kernelSize":           11,
    "dropout":              0.5,
})
configCNN6load = configCNN6.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configCNN7 = configGlobal.copy({
    "path":                 'configCNN7',
    "dataset":              'dataset32AugV3.hdf5',
    "split":                20,
    "batchSize":            256,
    "model":                'modelCNN',
    "filters":              128,
    "dropout":              0.5,
    "pooling":              'avg',
})
configCNN7load = configCNN7.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinCNN1 = configGlobal.copy({
    "path":                 'configWinCNN1',
    "dataset":              'dataset32AugW1.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32],
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "hiddenDim":            [16],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN1load = configWinCNN1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configWinCNN1loadReal31 = configWinCNN1.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.1.npz',
    "filePredAppendix":     "R3.1",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN1loadReal32 = configWinCNN1.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.2.npz',
    "filePredAppendix":     "R3.2",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN1loadReal33 = configWinCNN1.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.3.npz',
    "filePredAppendix":     "R3.3",
    "split":                None,
    "shuffleDataset":       False,
})


# ================================================
configTuneWinCNN1 = configGlobal.copy({
    "nonVerbose":           True,
    "useTune":              True,
    "path":                 'configTuneWinCNN1',
    "dataset":              'dataset32AugW1.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "model":                'modelCNNwin',
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "tuneConf":             {
        "batchSize":            hp.choice("batchSize", [8, 16, 32, 64, 128, 256]),
        "pooling":              hp.choice("pooling", ["max", "avg"]),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "hiddenDim":            [scope.int((hp.quniform("hiddenDim", 1, 512, 1)))],
        "filters":              [scope.int((hp.quniform("filters0", 1, 32, 1))), scope.int((hp.quniform("filters1", 1, 32, 1))), scope.int((hp.quniform("filters2", 1, 32, 1)))],
    }
})
configTuneWinCNN1load = configTuneWinCNN1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinCNN2 = configGlobal.copy({
    "path":                 'configWinCNN2',
    "dataset":              'dataset32AugW1.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32,64,128,256],
    "convKernel":           [6,3,3,4,3,3],
    "poolKernel":           [2,2,2,2,2,2],
    "hiddenDim":            [128],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN2load = configWinCNN2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configWinCNN2loadReal = configWinCNN2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-Ra.npz',
    "filePredAppendix":     "Ra",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2loadReal31 = configWinCNN2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.1.npz',
    "filePredAppendix":     "R3.1",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2loadReal32 = configWinCNN2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.2.npz',
    "filePredAppendix":     "R3.2",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2loadReal33 = configWinCNN2.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.3.npz',
    "filePredAppendix":     "R3.3",
    "split":                None,
    "shuffleDataset":       False,
})

# ================================================
# not runned
configTuneWinCNN2 = configGlobal.copy({
    "nonVerbose":           True,
    "useTune":              True,
    "path":                 'configTuneWinCNN2',
    "dataset":              'dataset32AugW1.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "model":                'modelCNNwin',
    "convKernel":           [6,3,3,4,3,3],
    "poolKernel":           [2,2,2,2,2,2],
    "tuneConf":             {
        "batchSize":            hp.choice("batchSize", [8, 16, 32, 64, 128, 256]),
        "pooling":              hp.choice("pooling", ["max", "avg"]),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "hiddenDim":            [scope.int((hp.quniform("hiddenDim", 1, 512, 1)))],
        "filters":              [scope.int((hp.quniform("filters{}".format(i), 1, 256, 1))) for i in range(6)],
    }
})

# ================================================
configWinCNN1W2 = configGlobal.copy({
    "path":                 'configWinCNN1W2',
    "dataset":              'dataset32AugW2.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32],
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "hiddenDim":            [16],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN1W2load = configWinCNN1W2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinCNN2W2 = configGlobal.copy({
    "path":                 'configWinCNN2W2',
    "dataset":              'dataset32AugW2.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32,64,128,256],
    "convKernel":           [6,3,3,4,3,3],
    "poolKernel":           [2,2,2,2,2,2],
    "hiddenDim":            [128],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN2W2load = configWinCNN2W2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinCNN1W3 = configGlobal.copy({
    "path":                 'configWinCNN1W3',
    "dataset":              'dataset32AugW3.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32],
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "hiddenDim":            [16],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN1W3load = configWinCNN1W3.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinCNN2W3 = configGlobal.copy({
    "path":                 'configWinCNN2W3',
    "dataset":              'dataset32AugW3.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [8,16,32,64,128,256],
    "convKernel":           [6,3,3,4,3,3],
    "poolKernel":           [2,2,2,2,2,2],
    "hiddenDim":            [128],
    "dropout":              0.,
    "pooling":              'max',
})
configWinCNN2W3load = configWinCNN2W3.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})
configWinCNN2W3loadReal = configWinCNN2W3.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-Ra.npz',
    "filePredAppendix":     "Ra",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2W3loadReal31 = configWinCNN2W3.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.1.npz',
    "filePredAppendix":     "R3.1",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2W3loadReal32 = configWinCNN2W3.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.2.npz',
    "filePredAppendix":     "R3.2",
    "split":                None,
    "shuffleDataset":       False,
})
configWinCNN2W3loadReal33 = configWinCNN2W3.copy({
    "modelLoad":            'best.pt',
    "dataset":              'dataset32-REAL-R3.3.npz',
    "filePredAppendix":     "R3.3",
    "split":                None,
    "shuffleDataset":       False,
})

# ================================================
configTuneWinCNN2W3 = configGlobal.copy({
    "nonVerbose":           True,
    "useTune":              True,
    "path":                 'configTuneWinCNN2W3',
    "dataset":              'dataset32AugW3.hdf5',
    "split":                20,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "model":                'modelCNNwin',
    "convKernel":           [6,3,3,4,3,3],
    "poolKernel":           [2,2,2,2,2,2],
    "tuneConf":             {
        "batchSize":            hp.choice("batchSize", [8, 16, 32, 64, 128, 256]),
        "pooling":              hp.choice("pooling", ["max", "avg"]),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "hiddenDim":            [scope.int((hp.quniform("hiddenDim", 1, 512, 1)))],
        "filters":              [scope.int((hp.quniform("filters{}".format(i), 1, 256, 1))) for i in range(6)],
    }
})

# ================================================ real validation
configWinRvCNN1 = configGlobal.copy({
    "path":                 'configWinRvCNN1',
    "shuffleDataset":       False,
    "customValidTest":      {
        "train":    'dataset32AugW1.hdf5',
        "valid":    'dataset32-REAL-R4.2.npz',
        "test":     'dataset32-REAL-R4.3.npz',
    },
    "split":                None,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            32,
    "model":                'modelCNNwin',
    "filters":              [4,4,4],
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "hiddenDim":            [16],
    "dropout":              0.5,
    "pooling":              'avg',
    "logEveryBatch":        True,
})
configWinRvCNN1load = configWinRvCNN1.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinRvCNN2 = configGlobal.copy({
    "path":                 'configWinRvCNN2',
    "shuffleDataset":       False,
    "customValidTest":      {
        "train":    'dataset32AugW4.hdf5',
        "valid":    'dataset32-REAL-R4.2.npz',
        "test":     'dataset32-REAL-R4.3.npz',
    },
    "split":                None,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            16,
    "model":                'modelCNNwin',
    "filters":              [1,2,4],
    "convKernel":           [4,4,4],
    "poolKernel":           [3,3,3],
    "hiddenDim":            [16],
    "dropout":              0.2,
    "pooling":              'avg',
    "logEveryBatch":        True,
})
configWinRvCNN2load = configWinRvCNN2.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})

# ================================================
configWinRvMLP = configGlobal.copy({
    "path":                 'configWinRvMLP',
    "shuffleDataset":       False,
    "customValidTest":      {
        "train":    'dataset32AugW4.hdf5',
        "valid":    'dataset32-REAL-R4.2.npz',
        "test":     'dataset32-REAL-R4.3.npz',
    },
    "split":                None,
    "dimX":                 201,
    "dimY":                 3,
    "inputB":               True,
    "batchSize":            1,
    "model":                'modelMLPwin',
    "activation":           'relu',
    "finalActivation":      'sigmoid',
    "hiddenLayers":         0,
    # "hiddenDim":            950,
    "dropout":              0.2,
    "logEveryBatch":        True,
})
configWinRvMLPload = configWinRvMLP.copy({
    "modelLoad":            'best.pt',
    "shuffleDataset":       False,
})


# ================================================
configMC1 = configGlobal.copy({
    "path":                 'configMC1',
    "taskType":             "regressionS",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,9),
    "rangeCollapse":        (0,3),
    "bestKey":              "e",
    "dimX":                 9*(101+51+41),
    "hiddenDim":            950,
    "modelLoad":            'best.pt',
})

configMC1test = configMC1.copy({
    "force":                True,
    "path":                 'configMC1test',
})

configMC1all = defaultdict(lambda: defaultdict(Conf))
configMC1allContinue = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC1all[n][c] = configGlobal.copy({
            "path":                 'configMC1all-{}-{}'.format(n,c),
            "nonVerbose":           True,
            "taskType":             "regressionS",
            "datasetType":          'multiCollapse',
            "multiCollapseMinMax":  multiCollapseMinMax1,
            "multiCollapseRanges":  multiCollapseRanges1,
            "multiCollapseIgnoreB": False,
            "dataset":              'dataset32MC.npz',
            "split":                'dataset32MC-split.npz',
            "rangeNN":              (0,n+1),
            "rangeCollapse":        (0,c+1),
            "bestKey":              "e",
            "dimX":                 (n+1)*(101 if c==0 else (101+51 if c==1 else 101+51+41)),
            "hiddenDim":            950,
            "modelLoad":            'best.pt',
        })

        configMC1allContinue[n][c] = configMC1all[n][c].copy({
            "modelLoad":            'last.pt',
            "startEpoch":           -1,
            "epochs":               900,
        })




# ================================================
#good for linear regression

configMC2all = defaultdict(lambda: defaultdict(Conf))
configMC2allContinue = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2all[n][c] = configMC1all[n][c].copy({
            "path":                 'configMC2all-{}-{}'.format(n,c),
            "normalizeY":           False,
            "hiddenDim":            2048,
            "finalActivation":      'none',
            "dropout":              0.5,
            "weightDecay":          0.0001,
            "batchSize":            8,
        })

        configMC2allContinue[n][c] = configMC2all[n][c].copy({
            "modelLoad":            'last.pt',
            "startEpoch":           -1,
            "epochs":               900,
        })

configMC2Nall = defaultdict(lambda: defaultdict(Conf))
configMC2NallContinue = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2Nall[n][c] = configMC2all[n][c].copy({
            "path":                 'configMC2Nall-{}-{}'.format(n,c),
            "dataset":              'dataset32MC-n.npz',
        })

configMC2allP = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2allP[n][c] = configMC2all[n][c].copy({
            "path":                 'configMC2allP-{}-{}'.format(n,c),
            "polyDegree":           2,
        })

configMC2NallP = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2NallP[n][c] = configMC2Nall[n][c].copy({
            "path":                 'configMC2NallP-{}-{}'.format(n,c),
            "polyDegree":           2,
        })

configMC2allMP = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2allMP[n][c] = configMC2all[n][c].copy({
            "path":                 'configMC2allMP-{}-{}'.format(n,c),
            "maxPolyDegree":        5,
        })

configMC2NallMP = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configMC2NallMP[n][c] = configMC2Nall[n][c].copy({
            "path":                 'configMC2NallMP-{}-{}'.format(n,c),
            "maxPolyDegree":        5,
        })


configCompareLinearRegression = Conf({
    "path":                 'configCompareLinearRegression',
    "normalizeY":           False,
    "dataset":              'dataset32MC-n.npz',
    "split":                'dataset32MC-split.npz',
    "compareDataset":       'real/XY8_N8_528G_from-old-data.dat',
    "compareParameters":    [0.004,0.48,528,0.0062],
    "rangeNN":              (2,3),
    "rangeCollapse":        (0,1),
})

configLinearRegressionMCR = defaultdict(lambda: defaultdict(Conf))
configLinearRegressionMCRn = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configLinearRegressionMCR[n][c] = Conf({
            "path":                 'configLinearRegressionMCR-{}-{}'.format(n,c),
            "normalizeY":           False,
            "dataset":              'dataset32MCR.npz',
            "split":                'dataset32MCR-split.npz',
            "rangeNN":              (0,n+1),
            "rangeCollapse":        (0,c+1),
        })
        configLinearRegressionMCRn[n][c] = configLinearRegressionMCR[n][c].copy({
            "path":                 'configLinearRegressionMCRn-{}-{}'.format(n,c),
            "dataset":              'dataset32MCR-n.npz',
        })

# ================================================
configTuneMC1 = configGlobal.copy({
    "path":                 'configTuneMC1',
    "taskType":             "regressionS",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,9),
    "rangeCollapse":        (0,3),
    "bestKey":              "e",
    "dimX":                 9*(101+51+41),
    "modelLoad":            'best.pt',
    "normalizeY":           False,
    "finalActivation":      'none',
    "nonVerbose":           True,
    "useTune":              True,
    "tuneConf":             {
        "learningRate":         tune.choice([0.01, 0.001, 0.0001]),
        "weightDecay":          tune.choice([0, 0.00001, 0.0001, 0.001]),
        "batchSize":            tune.choice([8, 16, 32]),
        "dropout":              tune.choice([0, 0.5]),
        "hiddenLayers":         tune.choice([0, 1, 2, 4, 8, 16]),
        # "hiddenDim":            tune.sample_from(lambda spec: np.random.choice(np.arange(2, 1024+2, 2)) if spec.config.hiddenLayers != 0 else 0),
        "hiddenDim":            tune.choice(list(np.arange(2, 1024+2, 2))),
    }
})

configTuneMCRn1 = configTuneMC1.copy({
    "path":                 'configTuneMCRn1',
    "dataset":              'dataset32MCR-n.npz',
    "split":                'dataset32MCR-split.npz',
})

# to test max size (max 15 parallel)
configTuneMC1max = configGlobal.copy({
    "force":                True,
    "path":                 'configTuneMC1max',
    "taskType":             "regressionS",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,9),
    "rangeCollapse":        (0,3),
    "bestKey":              "e",
    "dimX":                 9*(101+51+41),
    "modelLoad":            'best.pt',
    "normalizeY":           False,
    "finalActivation":      'none',
    "batchSize":            32,
    "hiddenLayers":         16,
    "hiddenDim":            1024,
})


configTuneMC2 = configTuneMC1.copy({
    "path":                 'configTuneMC2',
    "taskType":             "regressionL2",
    "bestKey":              "mse",
})

configTuneMC2test = configTuneMC2.copy({
    "path":                 'configTuneMC2test',
})

configTuneMC1appo = configTuneMC1.copy({
    "path":                 'configTuneMC1appo',
    "modelLoad":            'best.pt',
    "nonVerbose":           False,
    "useTune":              False,
    "batchSize":            16,
    "learningRate":         0.0001,
    "weightDecay":          0,
    "dropout":              0,
    "hiddenLayers":         16,
    "hiddenDim":            66,
})

configTuneMC2appo = configTuneMC2.copy({
    "path":                 'configTuneMC2appo',
    "modelLoad":            'best.pt',
    "nonVerbose":           False,
    "useTune":              False,
    "batchSize":            8,
    "learningRate":         0.0001,
    "weightDecay":          0.00001,
    "dropout":              0,
    "hiddenLayers":         2,
    "hiddenDim":            780,
})

configTuneMC3 = configGlobal.copy({
    "path":                 'configTuneMC3',
    "taskType":             "regressionL2",
    "bestKey":              "mse",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,9),
    "rangeCollapse":        (0,3),
    "dimX":                 9*(101+51+41),
    "modelLoad":            'best.pt',
    "normalizeY":           False,
    "finalActivation":      'none',
    "hiddenLayers":         0,
    "hiddenDim":            0,
    "nonVerbose":           True,
    "useTune":              True,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-1, 1e-2, 1e-3, 1e-4]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
    }
})

configTuneMC3max = configTuneMC3.copy({
    "force":                True,
    "path":                 'configTuneMC3max',
    "nonVerbose":           False,
    "useTune":              False,
})




configTuneMClinearRegression = configGlobal.copy({
    "path":                 'configTuneMClinearRegression',
    "epochs":               10000,
    "taskType":             "regressionL2",
    "batchTraining":        True,
    "optimizer":            'sgd', #sgd or adam
    "learningRate":         1e-3,
    "weight_decay":         0,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,9),
    "rangeCollapse":        (0,3),
    "dimX":                 9*(101+51+41),
    "normalizeY":           False,
})

configTuneMClinearRegression2 = configTuneMClinearRegression.copy({
    "path":                 'configTuneMClinearRegression2',
    "optimizer":            'adam', #sgd or adam
})

configTuneMClinearRegression3 = configTuneMClinearRegression2.copy({
    "path":                 'configTuneMClinearRegression3',
    "taskType":             "regressionS",
    "epochs":               100,
    "batchTraining":        True,
    "optimizer":            'sgd', #sgd or adam
    "learningRate":         1e-5,
})

configTuneMClinearRegression4 = configTuneMClinearRegression3.copy({
    "path":                 'configTuneMClinearRegression4',
    "learningRate":         1e-6,
})

# ================================================ New comparative experiments
# for tuneRunPytorch and runLinearRegressionSingle
configComp1 = configGlobal.copy({
    "path":                 'configComp1',
    "nonVerbose":           True,
    "taskType":             "regressionS",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (2,6), #N in {8,16,24,32}
    "rangeCollapse":        (0,1), #first collapse
    "bestKey":              "e",
    "trackMetric":          "e",
    "dimX":                 4*101,
    "hiddenDim":            950,
    "modelLoad":            'best.pt',
    "finalActivation":      'none',
    "useTune":              True,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 2048),
    }
})

configComp1poly = configComp1.copy({
    "path":                 'configComp1poly',
    "polyDegree":           2,
})

configComp1l2 = configComp1.copy({
    "path":                 'configComp1l2',
    "taskType":             "regressionL2",
})

# for runLinearRegressionPytorch
configCompTestLR = configComp1.copy({
    "path":                 'configCompTestLR',
    "learningRate":         1e-4,
    "nonVerbose":           False,
    "batchTraining":        False,
})

# ================================================ with fixed B=528

configComp2 = configComp1.copy({
    "path":                 'configComp2',
    # "dataset":              'dataset32MCnoB-n025.npz',
    "dataset":              'dataset32MC.npz',
    "split":                'dataset32MC-split.npz',
    "rangeNN":              (0,3), #N in {1,4,8}
    "dimX":                 3*101,
})

configSticazzi = configGlobal.copy({
            "path":                 'configSticazzi',
            "nonVerbose":           True,
            "taskType":             "regressionS",
            "datasetType":          'multiCollapse',
            "multiCollapseMinMax":  multiCollapseMinMax1,
            "multiCollapseRanges":  multiCollapseRanges1,
            "multiCollapseIgnoreB": False,
            "dataset":              'dataset32MC.npz',
            "split":                'dataset32MC-split.npz',
            "rangeNN":              (0,3),
            "rangeCollapse":        (0,1),
            "bestKey":              "e",
            "dimX":                 3*101,
            "hiddenDim":            950,
            "modelLoad":            'best.pt',
        })

configCompB528_1 = configComp1.copy({
    "path":                 'configCompB528_1',
    "fixedB":               528,
    # "dataset":              'dataset32MCnoB-n025.npz',
    "dataset":              'dataset32MCnoB.npz',
    "split":                'dataset32MCnoB-split.npz',
    "dimY":                 3,
    "rangeNN":              (0,3), #N in {1,4,8}
    "dimX":                 3*101,
})

configCompB528_1l2 = configCompB528_1.copy({
    "path":                 'configCompB528_1l2',
    "taskType":             "regressionL2",
})

configCompB528_1poly = configCompB528_1.copy({
    "path":                 'configCompB528_1poly',
    "polyDegree":           2,
})

#run with python runLinearRegression.py ...
configCompB528all = defaultdict(lambda: defaultdict(Conf))
configCompB528allN01 = defaultdict(lambda: defaultdict(Conf))
configCompB528allN025 = defaultdict(lambda: defaultdict(Conf))
configCompB528allN05 = defaultdict(lambda: defaultdict(Conf))
for n in range(9):
    for c in range(3):
        configCompB528all[n][c] = configGlobal.copy({
            "path":                 'configCompB528all-{}-{}'.format(n,c),
            "fixedB":               528,
            "dataset":              'dataset32MCnoB.npz',
            "split":                'dataset32MCnoB-split.npz',
            # "dimY":                 3,
            # "nonVerbose":           True,
            # "taskType":             "regressionS",
            # "datasetType":          'multiCollapse',
            "rangeNN":              (0,n+1),
            "rangeCollapse":        (0,c+1),
            # "bestKey":              "e",
            # "dimX":                 (n+1)*(101 if c==0 else (101+51 if c==1 else 101+51+41)),
            # "modelLoad":            'best.pt',
            "normalizeY":           False,
            #
            # "trackMetric":          "e",
            # "finalActivation":      'none',
            # "useTune":              True,
            # "tuneConf":             {
            #     "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
            #     "dropout":              tune.choice([0, 0.5]),
            #     "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
            #     "batchSize":            tune.choice([2, 4, 8, 16, 32]),
            #     "hiddenLayers":         tune.lograndint(1, 32),
            #     "hiddenDim":            tune.lograndint(1, 2048),
            # }
        })

        configCompB528allN01[n][c] = configCompB528all[n][c].copy({
            "path":                 'configCompB528allN01-{}-{}'.format(n,c),
            "dataset":              'dataset32MCnoB-n01.npz',
        })
        configCompB528allN025[n][c] = configCompB528all[n][c].copy({
            "path":                 'configCompB528allN025-{}-{}'.format(n,c),
            "dataset":              'dataset32MCnoB-n025.npz',
        })
        configCompB528allN05[n][c] = configCompB528all[n][c].copy({
            "path":                 'configCompB528allN05-{}-{}'.format(n,c),
            "dataset":              'dataset32MCnoB-n05.npz',
        })

configCompB528allN05_last = configCompB528allN05[8][2].copy({
    "path":                 'configCompB528allN05_last',
})

configCompB528allN05_lastNN = configGlobal.copy({
    "path":                 'configCompB528allN05_lastNN',
    "fixedB":               528,
    "dataset":              'dataset32MCnoB-n05.npz',
    # "dataset":              'dataset32MCnoB.npz',
    "split":                'dataset32MCnoB-split.npz',
    "dimY":                 3,
    "nonVerbose":           True,
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "rangeNN":              (0,8+1),
    "rangeCollapse":        (0,2+1),
    "bestKey":              "e",
    "dimX":                 (8+1)*(101 if 2==0 else (101+51 if 2==1 else 101+51+41)),
    "modelLoad":            'best.pt',
    "trackMetric":          "e",
    "finalActivation":      'none',
    "useTune":              True,
    "tuneConf":             {
        "normalizeY":           tune.choice([True, False]),
        "taskType":             tune.choice(["regressionS", "regressionL2"]),
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 2048),
    }
})

configCompB528allN05_32aNN = configCompB528allN05_lastNN.copy({
    "path":                 'configCompB528allN05_32aNN',
    "rangeNN":              (0,5+1), #{1,4,8,16,24,32}
    "rangeCollapse":        (0,0+1), #{a}
    "dimX":                 (5+1)*(101 if 0==0 else (101+51 if 0==1 else 101+51+41)),
})

configCompB528allN05_32aRNN = configCompB528allN05_32aNN.copy({
    "path":                 'configCompB528allN05_32aRNN',
    "model":                'modelRNN',
    "dimX":                 6,
    "tuneConf":             {
        "normalizeY":           tune.choice([True, False]),
        "taskType":             tune.choice(["regressionS", "regressionL2"]),
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.5]),
        "inputDropout":         tune.sample_from(lambda spec: (0,spec.config.dropout)),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.choice([1, 2, 3, 4]),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    }
})

#not perfectly completed because changed code in the middle...
configCompB528allN05_32aRNN2 = configCompB528allN05_32aRNN.copy({
    "path":                 'configCompB528allN05_32aRNN2',
    "logCurves":            False,
    "tensorBoard":          False,
    "tuneConf":             {
        "normalizeY":           tune.choice([True, False]),
        "taskType":             tune.choice(["regressionS", "regressionL2"]),
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.5]),
        "inputDropout":         tune.sample_from(lambda spec: (0,spec.config.dropout)),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    }
})

configCompB528allN05_32aRNN3 = configCompB528allN05_32aRNN2.copy({
    "path":                 'configCompB528allN05_32aRNN3',
    "taskType":             "regressionS",
    "normalizeY":           False,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "inputDropout":         tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    }
})


# to build all the table (with 10 trials per point) (model optimized for N=56, k=c)
configCompB528allN05_tune = configGlobal.copy({
    "path":                 'configCompB528allN05_tune',
    "fixedB":               528,
    "dataset":              'dataset32MCnoB-n05.npz',
    # "dataset":              'dataset32MCnoB.npz',
    "split":                'dataset32MCnoB-split.npz',
    "dimY":                 3,
    "nonVerbose":           True,
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "bestKey":              "e",
    "modelLoad":            'best.pt',
    "trackMetric":          "e",
    "finalActivation":      'none',
    "dropout":              0,
    "learningRate":         1e-4,
    "normalizeY":           False,
    "taskType":             "regressionS",
    "weightDecay":          1e-6,
    "batchSize":            2,
    "hiddenLayers":         6,
    "hiddenDim":            125,
    "useTune":              True,
    "tuneHyperOpt":         False,
    "tuneNumSamples":       10,
    "tuneConf":             {
        "NN":                   tune.grid_search(list(range(9))),
        "NC":                   tune.grid_search(list(range(3))),
        # "rangeNN":              tune.grid_search([(0,n+1) for n in range(9)]),
        # "rangeCollapse":        tune.grid_search([(0,c+1) for c in range(3)]),
        "rangeNN":              tune.sample_from(lambda spec: (0,spec.config.NN+1)),
        "rangeCollapse":        tune.sample_from(lambda spec: (0,spec.config.NC+1)),
        "dimX":                 tune.sample_from(lambda spec: (spec.config.NN+1)*(101 if spec.config.NC==0 else (101+51 if spec.config.NC==1 else 101+51+41))),
    }
})

# to build all the table (with 10 trials per point) (model optimized for N=32, k=a)
configCompB528allN05_tune2 = configCompB528allN05_tune.copy({
    "path":                 'configCompB528allN05_tune2',
    "batchSize":            8,
    "hiddenDim":            1478,
    "hiddenLayers":         5,
})

# train with python tuneRunPytorch.py configCompB528allN05_2coll
# syntetic data plot with calculateTuneTable.py
configCompB528allN05_2coll = configGlobal.copy({
    "path":                 'configCompB528allN05_2coll',
    "fixedB":               528,
    "datasetExperimental":  'dataset32MCnoB-experimental.npz',
    "dataset":              'dataset32MCnoB-n05.npz',
    # "dataset":              'dataset32MCnoB.npz',
    "split":                'dataset32MCnoB-split.npz',
    "dimY":                 3,
    "nonVerbose":           True,
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax1,
    "multiCollapseRanges":  multiCollapseRanges1,
    "multiCollapseIgnoreB": False,
    "bestKey":              "e",
    "modelLoad":            'best.pt',
    "trackMetric":          "e",
    "finalActivation":      'none',
    "dropout":              0,
    "learningRate":         1e-4,
    "normalizeY":           False,
    "taskType":             "regressionS",
    "weightDecay":          1e-6,
    "batchSize":            8,
    "hiddenLayers":         5,
    "hiddenDim":            1478,
    "rangeCollapse":        (1,3),
    "useTune":              True,
    "tuneHyperOpt":         False,
    "tuneNumSamples":       10,
    "tuneConf":             {
        "NN":                   tune.grid_search(list(range(2,9))), #{8,16,24,32,40,48,56}
        "rangeNN":              tune.sample_from(lambda spec: (0,spec.config.NN+1)),
        "dimX":                 tune.sample_from(lambda spec: (spec.config.NN+1)*(51+41)),
    }
})

# run with python runLinearRegression.py configCompB528allN05_2collLR
configCompB528allN05_2collLR = defaultdict(lambda: defaultdict(Conf))
for n in range(2,9):
    for c in range(2,3): #only c=2 (to keep same script)
        configCompB528allN05_2collLR[n][c] = configGlobal.copy({
            "path":                 'configCompB528allN05_2collLR-{}-{}'.format(n,c),
            "fixedB":               528,
            "datasetExperimental":  'dataset32MCnoB-experimental.npz',
            "dataset":              'dataset32MCnoB-n05.npz',
            "split":                'dataset32MCnoB-split.npz',
            "rangeNN":              (0,n+1),
            "rangeCollapse":        (1,c+1),
            "normalizeY":           False,
        })

# ================================================ New experiments with random dataset and new parameters
# for tuneRunPytorch and runLinearRegressionSingle
#FORGOT     "dimY": 3
configMCRnew1 = configGlobal.copy({
    "path":                 'configMCRnew1',
    "nonVerbose":           True,
    "taskType":             "regressionS",
    "datasetType":          'multiCollapse',
    "multiCollapseMinMax":  multiCollapseMinMax2,
    "multiCollapseRanges":  multiCollapseRanges2,
    # "multiCollapseIgnoreB": False,
    "multiCollapseIgnoreB": True,
    "fixedB":               403.2,
    "dataset":              'dataset32MCRnew.npz',
    "split":                'dataset32MCRnew-split.npz',
    "datasetExperimental":  'dataset32MCnoB-experimental2021.npz',
    "rangeNN":              (0,7), #N in {1, 8, 16, 24, 32, 40, 48}
    "rangeCollapse":        (0,2), #both collapses
    "bestKey":              "e",
    "trackMetric":          "e",
    "dimX":                 7*50,
    "modelLoad":            'best.pt',
    "finalActivation":      'none',
    "normalizeY":           False,
    "useTune":              True,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "inputDropout":         tune.choice([0, 0.2, 0.5]),                 #useless, MLP not RNN!
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),               #useless, MLP not RNN!
    },
})

#runned on qdab
configMCRnew2 = configMCRnew1.copy({
    "path":                 'configMCRnew2',
    "dataset":              'dataset32MCRnew1000.npz',
    "split":                'dataset32MCRnew1000-split.npz',
    "tuneParallelProc":     50,
    "tuneGrace":            1,
})

#runned on qdab (~ 2 days) and partially runned on alien (for nan prob.)
configMCRnew3 = configMCRnew1.copy({
    "path":                 'configMCRnew3',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
    "tuneParallelProc":     10,
    "tuneGrace":            1,
})

#----------
#runned on alien
configMCRnew2b = configMCRnew2.copy({
    "path":                 'configMCRnew2b',
    "tuneParallelProc":     10,
    "tuneGrace":            1,
    "rangeNN":              (0,2), #N in {1, 8}
    "dimX":                 2*50,
})

#----------
#runned on alien
configMCRnew2c = configMCRnew2.copy({
    "path":                 'configMCRnew2c',
    "tuneParallelProc":     10,
    "tuneGrace":            1,
    "rangeNN":              (0,1), #N in {1}
    "dimX":                 50,
})

#----------------- optimized with 1000 samples, trained with 10000
#for runPytorch
configMCRnew2_10000 = configMCRnew2.copy({
    "path":                 'configMCRnew2_10000',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
    "nonVerbose":           False,
    "useTune":              False,
    "learningRate":         0.0001,
    "dropout":              0,
    "inputDropout":         0.2,
    "weightDecay":          1e-05,
    "batchSize":            8,
    "hiddenLayers":         9,
    "hiddenDim":            172,
    "rnnType":              'lstm',
})

configMCRnew2b_10000 = configMCRnew2b.copy({
    "path":                 'configMCRnew2b_10000',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
    "nonVerbose":           False,
    "useTune":              False,
    "learningRate":         0.0001,
    "dropout":              0,
    "inputDropout":         0.2,
    "weightDecay":          0.0001,
    "batchSize":            8,
    "hiddenLayers":         4,
    "hiddenDim":            723,
    "rnnType":              'lstm',
})

configMCRnew2c_10000 = configMCRnew2c.copy({
    "path":                 'configMCRnew2c_10000',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
    "nonVerbose":           False,
    "useTune":              False,
    "learningRate":         0.0001,
    "dropout":              0,
    "inputDropout":         0,
    "weightDecay":          0,
    "batchSize":            8,
    "hiddenLayers":         7,
    "hiddenDim":            142,
    "rnnType":              'lstm',
})

configMCRnew2b_debug = configMCRnew2b_10000.copy({
    "path":                 'configMCRnew2b_debug',
})
#-----

# interrupted
configMCRnew2_100k = configMCRnew2_10000.copy({
    "path":                 'configMCRnew2_100k',
    "dataset":              'dataset32MCRnew.npz',
    "split":                'dataset32MCRnew-split.npz',
})

# interrupted
configMCRnew2b_100k = configMCRnew2b_10000.copy({
    "path":                 'configMCRnew2b_100k',
    "dataset":              'dataset32MCRnew.npz',
    "split":                'dataset32MCRnew-split.npz',
})

# interrupted
configMCRnew2c_100k = configMCRnew2c_10000.copy({
    "path":                 'configMCRnew2c_100k',
    "dataset":              'dataset32MCRnew.npz',
    "split":                'dataset32MCRnew-split.npz',
})

#----- same but with L2 loss (also corrected useless tune params and dimY=3)

#don't work better than sloss
# predictions: [0.00154045 0.53307045 0.00142563]
configMCRnew2L2 = configMCRnew2.copy({
    "path":                 'configMCRnew2L2',
    "taskType":             "regressionL2",
    # "bestKey":              "mse",
    "dimY":                 3,
    "tuneParallelProc":     10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
})
configMCRnew2L2Test = configMCRnew2L2.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

#runned on alien (~12:40 h)
# predictions: [0.00164117, 0.8394941, 0.0050657]
configMCRnew2L2_10k = configMCRnew2L2.copy({
    "path":                 'configMCRnew2L2_10k',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
    "multiCollapseIgnoreB": True,
})
# pred: [0.00210727 0.589223   0.00611006]
configMCRnew2L2_10kTest = configMCRnew2L2_10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})


# not run
configMCRnew2bL2 = configMCRnew2b.copy({
    "path":                 'configMCRnew2bL2',
    "taskType":             "regressionL2",
    # "bestKey":              "mse",
    "dimY":                 3,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
})

# not run
configMCRnew2cL2 = configMCRnew2c.copy({
    "path":                 'configMCRnew2cL2',
    "taskType":             "regressionL2",
    # "bestKey":              "mse",
    "dimY":                 3,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
})

#--------------- with normalization
# work same as sloss (but without nan problems)
#runned on alien
configMCRnew2L2norm = configMCRnew2L2.copy({
    "path":                 'configMCRnew2L2norm',
    "normalizeY":           True,
})
configMCRnew2L2normTest = configMCRnew2L2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configCancellami = configMCRnew2L2.copy({
    "path":                 'configCancellami',
    "normalizeY":           True,
    "useTune": False,
    "learningRate":         1e-3,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            2,
    "hiddenLayers":         1,
    "hiddenDim":            1,

})

#runned on alien (~ 3 h)
configMCRnew2bL2norm = configMCRnew2bL2.copy({
    "path":                 'configMCRnew2bL2norm',
    "normalizeY":           True,
})
configMCRnew2bL2normTest = configMCRnew2bL2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

#runned on alien (~ 2:30 h)
configMCRnew2cL2norm = configMCRnew2cL2.copy({
    "path":                 'configMCRnew2cL2norm',
    "normalizeY":           True,
})
configMCRnew2cL2normTest = configMCRnew2cL2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

#complete with all remaining \bar{N}
# (series is: configMCRnew2cL2norm, configMCRnew2bL2norm, configMCRnew2b3L2norm, configMCRnew2b4L2norm, configMCRnew2b5L2norm, configMCRnew2b6L2norm, configMCRnew2L2norm
# all running on alien
configMCRnew2b3L2norm = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2b3L2norm',
    "rangeNN":              (0,3), #N in {1, 8, 16}
    "dimX":                 3*50,
})
configMCRnew2b3L2normTest = configMCRnew2b3L2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b4L2norm = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2b4L2norm',
    "rangeNN":              (0,4), #N in {1, 8, 16, 24}
    "dimX":                 4*50,
})
configMCRnew2b4L2normTest = configMCRnew2b4L2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b5L2norm = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2b5L2norm',
    "rangeNN":              (0,5), #N in {1, 8, 16, 24, 32}
    "dimX":                 5*50,
})
configMCRnew2b5L2normTest = configMCRnew2b5L2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b6L2norm = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2b6L2norm',
    "rangeNN":              (0,6), #N in {1, 8, 16, 24, 32, 40}
    "dimX":                 6*50,
})
configMCRnew2b6L2normTest = configMCRnew2b6L2norm.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

#------------ with 10k (compare also with configMCRnew3)
# runned on qdab (~ 20h)
configMCRnew2L2norm10k = configMCRnew2L2norm.copy({
    "path":                 'configMCRnew2L2norm10k',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
})
configMCRnew2L2norm10kTest = configMCRnew2L2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

# runned on qdab
configMCRnew2bL2norm10k = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2bL2norm10k',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
})
configMCRnew2bL2norm10kTest = configMCRnew2bL2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2cL2norm10k = configMCRnew2cL2norm.copy({
    "path":                 'configMCRnew2cL2norm10k',
    "dataset":              'dataset32MCRnew10000.npz',
    "split":                'dataset32MCRnew10000-split.npz',
})
configMCRnew2cL2norm10kTest = configMCRnew2cL2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})


#complete with all remaining \bar{N}
# (series is: configMCRnew2cL2norm10k, configMCRnew2bL2norm10k, configMCRnew2b3L2norm10k, configMCRnew2b4L2norm10k, configMCRnew2b5L2norm10k, configMCRnew2b6L2norm10k, configMCRnew2L2norm10k
# to run
configMCRnew2b3L2norm10k = configMCRnew2bL2norm10k.copy({
    "path":                 'configMCRnew2b3L2norm10k',
    "rangeNN":              (0,3), #N in {1, 8, 16}
    "dimX":                 3*50,
})
configMCRnew2b3L2norm10kTest = configMCRnew2b3L2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b4L2norm10k = configMCRnew2bL2norm10k.copy({
    "path":                 'configMCRnew2b4L2norm10k',
    "rangeNN":              (0,4), #N in {1, 8, 16, 24}
    "dimX":                 4*50,
})
configMCRnew2b4L2norm10kTest = configMCRnew2b4L2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b5L2norm10k = configMCRnew2bL2norm10k.copy({
    "path":                 'configMCRnew2b5L2norm10k',
    "rangeNN":              (0,5), #N in {1, 8, 16, 24, 32}
    "dimX":                 5*50,
})
configMCRnew2b5L2norm10kTest = configMCRnew2b5L2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

configMCRnew2b6L2norm10k = configMCRnew2bL2norm10k.copy({
    "path":                 'configMCRnew2b6L2norm10k',
    "rangeNN":              (0,6), #N in {1, 8, 16, 24, 32, 40}
    "dimX":                 6*50,
})
configMCRnew2b6L2norm10kTest = configMCRnew2b6L2norm10k.copy({
    "datasetExperimental": 'dataset32MCnoB-experimental2021test.npz',
})

#--------------- with 1k and rnn

# runned on alien (~3:20h)
configMCRnew2L2normRNN = configMCRnew2L2norm.copy({
    "path":                 'configMCRnew2L2normRNN',
    "model":                'modelRNN',
    "dimX":                 7, # why not 50?
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "inputDropout":         tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    },
})

# runned on alien (gpu mem errors)
configMCRnew2bL2normRNN = configMCRnew2bL2norm.copy({
    "path":                 'configMCRnew2bL2normRNN',
    "model":                'modelRNN',
    "dimX":                 2, # why not 50?
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "inputDropout":         tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    },
})

# runned on alien (~3:30, gpu mem errors)
configMCRnew2cL2normRNN = configMCRnew2cL2norm.copy({
    "path":                 'configMCRnew2cL2normRNN',
    "model":                'modelRNN',
    "dimX":                 1, # why not 50?
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "inputDropout":         tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
        "rnnType":              tune.choice(['lstm', 'gru']),
    },
})

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Use {} configName".format(sys.argv[0]))
    else:
        # conf = getattr(sys.modules['configurations'], sys.argv[1])
        conf = eval(format(sys.argv[1]))
        conf.print()
