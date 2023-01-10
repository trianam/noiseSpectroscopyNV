#     funPlot.py
#     Collect the function used to plot the learning curves.
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

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import sys
import os
import configurations
import warnings
from ray.tune import Analysis
import funPytorch
import torch
import loaders

plt.rcParams['figure.dpi'] = 200

def plot(configs, sets='test', save=False, colorsFirst=True, title="", limits=None, plotMetric=False):
    keyLoss = 'loss'

    lineStyles = ['solid', 'dashed', 'dotted', 'dashdot']
    lineColors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if not type(configs) is list:
        configs = [configs]

    if not type(sets) is list:
        sets = [sets]

    if len(configs)*len(sets) > len(lineStyles)*len(lineColors):
        raise ValueError("Too many curves to plot, {} of max {}.".format(len(configs)*len(sets), len(lineStyles)*len(lineColors)))

    if plotMetric:
        fig = plt.figure(figsize=(8,9))
        #gs = fig.add_gridspec(2,1)
        #axLoss = fig.add_subplot(gs[0, 0])
        #axAcc = fig.add_subplot(gs[1, 0])
        axLoss = fig.add_axes([0.1, 0.53, 0.6, 0.42])
        axAcc = fig.add_axes([0.1, 0.05, 0.6, 0.42])
    else:
        fig = plt.figure(figsize=(8,5))
        axLoss = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    #fig, (axLoss, axAcc) = plt.subplots(2)
    metrics = []
    i = 0
    for set in sets:
        for config in configs:
            myConfig = getattr(sys.modules['configurations'], config)
            metrics.append(myConfig.trackMetric)

            if myConfig.useTune:
                #use tune to pick best in run
                analysis = Analysis(join("tuneOutput", myConfig.path))
                mode = ("max" if myConfig.bestSign == '>' else "min")
                #print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
                tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
                expPath = join(tunePath,'files', 'tensorBoard')
            else:
                expPath = join('files', myConfig.path, 'tensorBoard')

            # keyAcc = "{}_{}".format(metrics[-1], metrics[-1])
            keyAcc = "{}".format(metrics[-1])

            # metrics.append(getattr(sys.modules['configurations'], config).trackMetric)
            # keyAcc = "{}_{}".format(metrics[-1], metrics[-1])
            # expPath = join('files', getattr(sys.modules['configurations'], config).path, 'tensorBoard')
            try:
                keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
            except FileNotFoundError:
                warnings.warn("Configuration {} not present. Skipping.".format(config))
                continue

            points = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points:
                        points[k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[k][0].extend([v.step for v in ea.Scalars(k)])
                        points[k][1].extend([v.value for v in ea.Scalars(k)])

            if limits is None:
                valuesLoss = points[keyLoss]
                if plotMetric:
                    valuesAcc = points[keyAcc]
            else:
                valuesLoss = [points[keyLoss][i][limits[0]:limits[1]] for i in [0,1]]
                if plotMetric:
                    valuesAcc = [points[keyAcc][i][limits[0]:limits[1]] for i in [0,1]]

            linesLoss = axLoss.plot(valuesLoss[0], valuesLoss[1], label="{} {}".format(config, set))
            if colorsFirst:
                linesLoss[0].set_color(lineColors[i%len(lineColors)])
                linesLoss[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
            else:
                linesLoss[0].set_linestyle(lineStyles[i%len(lineStyles)])
                linesLoss[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])

            if plotMetric:
                linesAcc = axAcc.plot(valuesAcc[0], valuesAcc[1], label="{} {}".format(config, set))
                if colorsFirst:
                    linesAcc[0].set_color(lineColors[i%len(lineColors)])
                    linesAcc[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
                else:
                    linesAcc[0].set_linestyle(lineStyles[i%len(lineStyles)])
                    linesAcc[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])

            i += 1


    # axLoss.legend(loc='upper left', bbox_to_anchor=(1, 1),
    #          ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
    #axLoss.set_title(title)
    axLoss.set_xlabel("Epoch")
    axLoss.set_ylabel("Loss")

    # axLoss.set_xticks(np.arange(0, round(axLoss.get_xlim()[1])+10, 10))
    # axLoss.set_xticks(np.arange(round(axLoss.get_xlim()[0]), round(axLoss.get_xlim()[1])+1, 1), minor=True)
    # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.05, 0.05))
    # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.01, 0.01), minor=True)
    axLoss.grid(which='both')
    axLoss.grid(which='minor', alpha=0.2)
    axLoss.grid(which='major', alpha=0.5)

    fig.suptitle(title)
    handles, labels = axLoss.get_legend_handles_labels()

    if plotMetric:
        # axAcc.legend(loc='upper left', bbox_to_anchor=(1, 1),
        #           ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
        #axAcc.set_title(title)
        axAcc.set_xlabel("Epoch")
        axAcc.set_ylabel("Metric ({})".format(list(dict.fromkeys(metrics))))
        #axAcc.set_ylim(0.49,1.)

        #axAcc.set_xticks(np.arange(0, round(axAcc.get_xlim()[1])+10, 10))
        #axAcc.set_xticks(np.arange(round(axAcc.get_xlim()[0]), round(axAcc.get_xlim()[1])+1, 1), minor=True)
        # axAcc.set_yticks(np.arange(0.5, 1.01, 0.1))
        # axAcc.set_yticks(np.arange(0.49, 1.01, 0.01), minor=True)
        axAcc.grid(which='both')
        axAcc.grid(which='minor', alpha=0.2)
        axAcc.grid(which='major', alpha=0.5)

        fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.95), loc=2, borderaxespad=0.)
    else:
        fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.9), loc=2, borderaxespad=0.)

    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1),
    #              bbox_transform = plt.gcf().transFigure,
    #              ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)

    # plt.legend( handles, labels, loc = 'upper left', bbox_to_anchor = (0.9,-0.1,2,2),
    #         bbox_transform = plt.gcf().transFigure )

    # plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), bbox_transform=plt.gcf().transFigure)

    # fig.subplots_adjust(wspace=2, hspace=2,left=0,top=2,right=2,bottom=0)

    #fig.tight_layout()

    #fig.subplots_adjust(right=2)

    fig.show()

    if save:
        os.makedirs("img", exist_ok=True)
        fig.savefig('img/plot.pdf')#, bbox_inches = 'tight')#, pad_inches = 0)

    plt.close()


def printMetricsOld(configs, printAllConfigs=False):
    sets = ['train', 'valid', 'test']

    if not type(configs) is list:
        configs = [configs]

    bestConfig = None
    points = {}
    for config in configs:
        myConfig = getattr(sys.modules['configurations'], config)
        metric = myConfig.trackMetric
        mode = ("max" if myConfig.bestSign == '>' else "min")

        if myConfig.useTune:
            try:
                #use tune to pick best in run
                analysis = Analysis(join("tuneOutput", myConfig.path))
            except ValueError:
                warnings.warn("Configuration {} not present. Skipping.".format(config))
                continue
            print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
            tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
            # expPath = join(tunePath,'files','tensorBoard')
            expPath = join(tunePath,'files',config,'tensorBoard')
            print("best logdir: {}".format(expPath))
        else:
            expPath = join('files', config, 'tensorBoard')
        # keyAcc = "{}_{}".format(metric, metric)
        keyAcc = "{}".format(metric)
        try:
            keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
        except FileNotFoundError:
            warnings.warn("Configuration {} not present. Skipping.".format(config))
            continue

        points[config] = {}
        for set in sets:
            points[config][set] = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points[config][set]:
                        points[config][set][k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[config][set][k][0].extend([v.step for v in ea.Scalars(k)])
                        points[config][set][k][1].extend([v.value for v in ea.Scalars(k)])

        bestSign = myConfig.bestSign
        if bestSign == '>':
            bestI = np.argmax(points[config]['valid'][keyAcc][1]) #point where better metric
        else:
            bestI = np.argmin(points[config]['valid'][keyAcc][1]) #point where better metric

        thisConfig = {
            'name': config,
            'epoch': bestI+1, 
            'train': points[config]['train'][keyAcc][1][bestI], 
            'valid': points[config]['valid'][keyAcc][1][bestI], 
            'test': points[config]['test'][keyAcc][1][bestI],
        }
        if printAllConfigs:
            print("{} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(thisConfig['name'], thisConfig['epoch'], thisConfig['train'], thisConfig['valid'], thisConfig['test']))
            
        if bestConfig is None or (bestSign == '>' and thisConfig['valid']>bestConfig['valid']) or (bestSign == '<' and thisConfig['valid']<bestConfig['valid']):
            bestConfig = thisConfig    

    if not bestConfig is None:
        print("BEST ==== {} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(bestConfig['name'], bestConfig['epoch'], bestConfig['train'], bestConfig['valid'], bestConfig['test']))


def printTableOld(config):
    conf = getattr(sys.modules['configurations'], config)

    basePath = os.path.join("tuneOutput", conf.path)
    analysis = Analysis(basePath)
    configsDict = analysis.get_all_configs()
    metric = conf.trackMetric
    # keyAcc = "{}_{}".format(metric, metric)
    keyAcc = "{}".format(metric)
    bestSign = conf.bestSign

    nns = [1,4,8,16,24,32,40,48,56]
    ks = ["a","b","c"]
    finalPoints = np.zeros((len(nns), len(ks)))

    for i_n,n in enumerate(nns):
        for i_c,c in enumerate(ks):
            points = {'valid':{}, 'test':{}} #concatenate all curves in points
            for expPath in configsDict:
                if configsDict[expPath]['NN'] == i_n and configsDict[expPath]['NC'] == i_c:
                    # tbPath = os.path.join(expPath,'files', conf.path, 'tensorBoard')
                    tbPath = os.path.join(expPath,'files', 'curves')
                    keys = [f for f in os.listdir(tbPath) if not os.path.isfile(os.path.join(tbPath, f))]
                    for set in ['valid', 'test']:
                        for k in keys:
                            eventPathPart = os.path.join(tbPath, k, set)
                            # for runPath in sorted([f for f in os.listdir(eventPathPart) if os.path.isfile(os.path.join(eventPathPart, f))]):
                            #     eventPath = os.path.join(eventPathPart, runPath)
                            #     ea = event_accumulator.EventAccumulator(eventPath)
                            #     ea.Reload()
                            #     if k in ea.Tags()['scalars']: #for skipping pending/unfinished experiments
                            #         if not k in points[set]:
                            #             points[set][k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                            #         else:
                            #             points[set][k][0].extend([v.step for v in ea.Scalars(k)])
                            #             points[set][k][1].extend([v.value for v in ea.Scalars(k)])

                            if not k in points[set]:
                                points[set][k] = list(np.loadtxt(os.path.join(eventPathPart, "points.dat")))
                            else:
                                points[set][k].extend(list(np.loadtxt(os.path.join(eventPathPart, "points.dat"))))

                            # if np.isnan(np.array(points[set][k])).any():
                            #     print(eventPathPart)

            # if bestSign == '>':
            #     bestI = np.argmax(points['valid'][keyAcc][1]) #point where better metric
            # else:
            #     bestI = np.argmin(points['valid'][keyAcc][1]) #point where better metric
            #
            # base,exp = np.format_float_scientific(points['test'][keyAcc][1][bestI], 3, exp_digits=1).split('e')

            if bestSign == '>':
                bestI = np.nanargmax(np.array(points['valid'][keyAcc])) #point where better metric
            else:
                bestI = np.nanargmin(np.array(points['valid'][keyAcc])) #point where better metric

            finalPoints[i_n][i_c] = points['test'][keyAcc][bestI]

    print("===================================== List")
    for i_n,n in enumerate(nns):
        for i_c,c in enumerate([1,2,3]):
            print("{} {} {}".format(n, c, finalPoints[i_n][i_c]))
        print("")

    print("===================================== Table")
    print("$L$ & " + " & ".join(["$k={}$".format(k) for k in ks]) + "\\\\")
    print("\\hline\\\\")
    for i_n,n in enumerate(nns):
        print("$\\overline{{N}}={}$".format(n), end='')
        for i_c,c in enumerate(ks):
            base,exp = np.format_float_scientific(finalPoints[i_n][i_c], 3, exp_digits=1).split('e')
            print("& ${}\\times 10^{{{}}}$".format(base, exp), end='')
        print("\\\\")


def printMetrics(config):
    myConfig = getattr(sys.modules['configurations'], config)
    argminmax = np.argmax if myConfig.bestSign == '>' else np.argmin
    lessgreat = lambda new, old: ((new>old) if myConfig.bestSign == '>' else (new<old))

    try:
        #use tune to pick best in run
        analysis = Analysis(join("tuneOutput", myConfig.path))
    except ValueError:
        raise ValueError("Configuration {} not present.".format(config))

    trials = analysis.fetch_trial_dataframes()
    configs = analysis.get_all_configs()

    validMetric = (0 if myConfig.bestSign == '>' else np.inf)
    testMetric = np.nan
    trainMetric = np.nan
    epoch = np.nan
    key = ""
    for k in trials:
        trial = trials[k]
        if "/".join(["valid",myConfig.bestKey]) in trial.columns:
            bestIndex = argminmax(trial["/".join(["valid",myConfig.bestKey])])
            if not bestIndex == -1 and lessgreat(trial["/".join(["valid",myConfig.bestKey])][bestIndex], validMetric): # not all nan and the minimum better than previous trials
                validMetric = trial["/".join(["valid",myConfig.bestKey])][bestIndex]
                testMetric = trial["/".join(["test",myConfig.bestKey])][bestIndex]
                trainMetric = trial["/".join(["train",myConfig.bestKey])][bestIndex]
                epoch = bestIndex
                key = k
        else:
            print("{} not present in {}".format("/".join(["valid",myConfig.bestKey]), k))

    print("BEST ==== {}\n\t (epoch {}) {}:\ttrain {:.3};\tvalid {:.3};\ttest {:.3}\n\t{}".format(key, epoch, myConfig.bestKey, trainMetric, validMetric, testMetric, configs[key]))




def printTable(config):
    myConfig = getattr(sys.modules['configurations'], config)

    try:
        analysis = Analysis(join("tuneOutput", myConfig.path))
    except ValueError:
        raise ValueError("Configuration {} not present.".format(config))

    trials = analysis.fetch_trial_dataframes()
    configsDict = analysis.get_all_configs()

    nns = np.array([1,4,8,16,24,32,40,48,56])
    inns = np.unique([configsDict[k]['NN'] for k in configsDict])    #take only explored configurations

    finalPoints = np.zeros(len(nns))

    for i_n in inns:
        points = {'valid':[], 'test':[]} #concatenate all curves in points
        for expPath in configsDict:
            if configsDict[expPath]['NN'] == i_n:
                for set in ['valid', 'test']:
                    points[set].extend(list(trials[expPath][set+"/"+myConfig.trackMetric]))

        if myConfig.bestSign == '>':
            bestI = np.nanargmax(np.array(points['valid'])) #point where better metric
        else:
            bestI = np.nanargmin(np.array(points['valid'])) #point where better metric

        finalPoints[i_n] = points['test'][bestI]

    print("===================================== List")
    for i_n in inns:
        n = nns[i_n]
        print("{} {}".format(n, finalPoints[i_n]))


def printTableExperimental(config):
    myConfig = getattr(sys.modules['configurations'], config)

    try:
        analysis = Analysis(join("tuneOutput", myConfig.path))
    except ValueError:
        raise ValueError("Configuration {} not present.".format(config))

    trials = analysis.fetch_trial_dataframes()
    configsDict = analysis.get_all_configs()

    nns = np.array([1,4,8,16,24,32,40,48,56])
    inns = np.unique([configsDict[k]['NN'] for k in configsDict])    #take only explored configurations

    finalPoints = np.zeros(len(nns))

    for i_n in inns:
        if myConfig.bestSign == '>':
            bestI = - np.infty
        else:
            bestI = np.infty

        bestConf = ""
        for expPath in configsDict:
            if configsDict[expPath]['NN'] == i_n:
                if myConfig.bestSign == '>':
                    currI = np.nanargmax(trials[expPath]["valid/"+myConfig.trackMetric])
                    if currI > bestI:
                        bestI = currI
                        bestConf = expPath
                else:
                    currI = np.nanargmin(trials[expPath]["valid/"+myConfig.trackMetric])
                    if currI < bestI:
                        bestI = currI
                        bestConf = expPath

        adhocConf = myConfig.copy({
            "rangeNN":      (0, i_n+1),
            "dimX":         (i_n+1) * (51+41),
        })
        device = "cpu"
        fileToLoad = os.path.join(bestConf, "files", "models", "best.pt")
        model, _ = funPytorch.makeModel(adhocConf, device)

        checkpoint = torch.load(fileToLoad, map_location=torch.device('cpu'))#map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        dataloader, _ = loaders.multiCollapseExperimental(adhocConf)

        loss, metrics = funPytorch.evaluate(adhocConf, model, dataloader)
        finalPoints[i_n] = metrics[adhocConf.trackMetric]


    print("===================================== List")
    for i_n in inns:
        n = nns[i_n]
        print("{} {}".format(n, finalPoints[i_n]))
