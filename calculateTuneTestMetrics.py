#!/usr/bin/env python
# coding: utf-8

#     calculateTuneTestMetrics.py
#     Reports all the results after the training for the best models in the considered categories.
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

from funPlot import printMetricsOld
from funPlot import printMetrics
from itertools import product
import numpy as np

# print("========================================================================================================== configTuneWinCNN1")
# printMetrics('configTuneWinCNN1')
# print("========================================================================================================== configTuneWinCNN2W3")
# printMetrics('configTuneWinCNN2W3')

# print("========================================================================================================== configComp1")
# printMetrics('configComp1')
# print("========================================================================================================== configComp1l2")
# printMetrics('configComp1l2')

# print("========================================================================================================== configCompB528allN05_lastNN")
# printMetricsOld('configCompB528allN05_lastNN')

# print("========================================================================================================== configCompB528allN05_32aNN")
# printMetricsOld('configCompB528allN05_32aNN')

# print("========================================================================================================== configCompB528allN05_32aRNN2")
# printMetrics('configCompB528allN05_32aRNN2')

# print("========================================================================================================== configCompB528allN05_32aRNN3")
# printMetrics('configCompB528allN05_32aRNN3')

# new experiments
print("========================================================================================================== configMCRnew2")
printMetrics('configMCRnew2')

print("========================================================================================================== configMCRnew2b")
printMetrics('configMCRnew2b')

print("========================================================================================================== configMCRnew2c")
printMetrics('configMCRnew2c')

# print("========================================================================================================== configMCRnew3")
# printMetrics('configMCRnew3')

print("========================================================================================================== configMCRnew2L2")
printMetrics('configMCRnew2L2')

print("========================================================================================================== configMCRnew2L2norm")
printMetrics('configMCRnew2L2norm')

print("========================================================================================================== configMCRnew2bL2norm")
printMetrics('configMCRnew2bL2norm')

print("========================================================================================================== configMCRnew2cL2norm")
printMetrics('configMCRnew2cL2norm')

print("========================================================================================================== configMCRnew2L2normRNN")
printMetrics('configMCRnew2L2normRNN')

print("========================================================================================================== configMCRnew2bL2normRNN")
printMetrics('configMCRnew2bL2normRNN')

print("========================================================================================================== configMCRnew2cL2normRNN")
printMetrics('configMCRnew2cL2normRNN')
