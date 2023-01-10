import os
from pathlib import Path
import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sl
import sklearn.metrics
import pickle
import time
import math
import warnings
from datetime import datetime

import torch
import torch.nn as nn

import torchsummary
from tensorboardX import SummaryWriter

from collections import defaultdict

import loaders
import importlib

from ray import tune
from ray.tune import Analysis


class MRELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yhat,y):
        return torch.mean(torch.abs(y - yhat) / y)
        # return torch.mean((y - yhat)**2)

class SLoss(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf

        B_lims   = [520,536]
        W1_lims   = [0.004,0.009]

        self.g=1.0705e-3 # C-13 nuclear spin gyromagnetic ratio
        self.omega2 = 2*np.pi*torch.linspace(self.g*min(B_lims)-5*max(W1_lims), self.g*max(B_lims)+5*max(W1_lims),500).to(device)
        
        if conf.normalizeY:
            self.minMax = conf.multiCollapseMinMax



    def _funcGauss(self, x,y0, a,xc,w):
        return y0+a*torch.exp(-0.5*((x-2*np.pi*xc)/(2*np.pi*w))**2) #I included a couple of 2*np.pi to convert \nu->

    def _funcNoise(self, x,y0,a1,x1,w1): # ,a2,x2,w2 ,a3,x3,w3 ,a4,x4,w4):
        return y0 + self._funcGauss(x,0,a1,x1,w1) #+ funcGauss(x,0,a2,x2,w2) + funcGauss(x,0,a3,x3,w3) + funcGauss(x,0,a4,x4,w4)

    def forward(self,yhatBatch,yBatch):
        # Y0, A, B, W1

        error = 0
        for yhat,y in zip(yhatBatch, yBatch):
            if self.conf.has("fixedB"):
                Y0 = y[0]
                Y0_hat = yhat[0]
                A = y[1]
                A_hat = yhat[1]
                B = self.conf.fixedB
                B_hat = self.conf.fixedB
                W1 = y[2]
                W1_hat = yhat[2]
            else:
                Y0 = y[0]
                Y0_hat = yhat[0]
                A = y[1]
                A_hat = yhat[1]
                B = y[2]
                B_hat = yhat[2]
                W1 = y[3]
                W1_hat = yhat[3]
                
            if self.conf.normalizeY: # denormalize in that case
                Y0 *= self.minMax[0][1] - self.minMax[0][0]
                Y0 += self.minMax[0][0]
                Y0_hat *= self.minMax[0][1] - self.minMax[0][0]
                Y0_hat += self.minMax[0][0]

                A *= self.minMax[1][1] - self.minMax[1][0]
                A += self.minMax[1][0]
                A_hat *= self.minMax[1][1] - self.minMax[1][0]
                A_hat += self.minMax[1][0]

                if not self.conf.has("fixedB"):
                    B *= self.minMax[2][1] - self.minMax[2][0]
                    B += self.minMax[2][0]
                    B_hat *= self.minMax[2][1] - self.minMax[2][0]
                    B_hat += self.minMax[2][0]

                W1 *= self.minMax[3][1] - self.minMax[3][0]
                W1 += self.minMax[3][0]
                W1_hat *= self.minMax[3][1] - self.minMax[3][0]
                W1_hat += self.minMax[3][0]

            vl = B_hat*self.g # B*\gamma [MHZ]
            para_A = [0.0, A_hat, vl, W1_hat] # [offset, amplitude, center, width] All in MHz

            vl = B*self.g # B*\gamma [MHZ]
            para_B = [0.0, A, vl, W1] # [offset, amplitude, center, width] All in MHz

            error += abs(Y0_hat - Y0)*(8.5-0.001) + sum(abs(self._funcNoise(self.omega2,*para_B)-self._funcNoise(self.omega2,*para_A)))*(self.omega2[1]-self.omega2[0])

        return error / len(yBatch)


class MyLoss(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf

        if conf.taskType == "classification":
            self.lossFun = nn.CrossEntropyLoss()
        elif conf.taskType == "prediction":
            self.lossFun = nn.KLDivLoss(reduction='batchmean')
        elif conf.taskType == "regressionL1":
            self.lossFun = nn.L1Loss()
        elif conf.taskType == "regressionL2":
            self.lossFun = nn.MSELoss()
        elif conf.taskType == "regressionMRE":
            self.lossFun = MRELoss()
        elif conf.taskType == "regressionS":
            self.lossFun = SLoss(conf, device)
        else:
            raise ValueError("taskType {} not valid".format(conf.taskType))

    def forward(self, yhatBatch, yBatch):
        return self.lossFun(yhatBatch, yBatch)

def filesPath(conf):
    if conf.has("runningPredictions") and conf.runningPredictions and conf.has("useTune") and conf.useTune: #load best configuration model
        mode = ("max" if conf.bestSign == '>' else "min")

        #use tune to pick best in run
        analysis = Analysis(os.path.join("tuneOutput", conf.path))
        try:
            tunePath = analysis.get_best_logdir(metric="/".join(["valid",conf.bestKey]), mode=mode)
        except KeyError:  #try to ignore faulty rows
            trialDF = analysis.trial_dataframes
            bestMetric = np.inf if mode=='min' else -np.inf
            for i in range(len(list(trialDF.keys()))):
                try:
                    if mode == 'min':
                        currMetric = min(trialDF[list(trialDF.keys())[i]]["/".join(["valid", conf.bestKey])])
                        if currMetric < bestMetric:
                            bestMetric = currMetric
                            tunePath = list(trialDF.keys())[i]
                    else:
                        currMetric = max(trialDF[list(trialDF.keys())[i]]["/".join(["valid", conf.bestKey])])
                        if currMetric > bestMetric:
                            bestMetric = currMetric
                            tunePath = list(trialDF.keys())[i]
                except KeyError:
                    print("Skipped: {}".format(list(trialDF.keys())[i]))

        return os.path.join(tunePath, "files")
        # return "files"
    elif conf.has("useTune") and conf.useTune:
        return "files"
    else:
        return os.path.join("files", conf.path)

def makeModel(conf, device):
    modelPackage = importlib.import_module("models."+conf.model)
    if conf.has("runningPredictions") and conf.runningPredictions and conf.has("useTune") and conf.useTune: #use tune to pick best in run and update conf with best hyperparameters
        mode = ("max" if conf.bestSign == '>' else "min")
        analysis = Analysis(os.path.join("tuneOutput", conf.path))
        try:
            conf = conf.copy(analysis.get_best_config(metric="/".join(["valid",conf.bestKey]), mode=mode))
        except KeyError:  #try to ignore faulty rows
            trialDF = analysis.trial_dataframes
            bestMetric = np.inf if mode=='min' else -np.inf
            for i in range(len(list(trialDF.keys()))):
                try:
                    if mode == 'min':
                        currMetric = min(trialDF[list(trialDF.keys())[i]]["/".join(["valid", conf.bestKey])])
                        if currMetric < bestMetric:
                            bestMetric = currMetric
                            tunePath = list(trialDF.keys())[i]
                    else:
                        currMetric = max(trialDF[list(trialDF.keys())[i]]["/".join(["valid", conf.bestKey])])
                        if currMetric > bestMetric:
                            bestMetric = currMetric
                            tunePath = list(trialDF.keys())[i]
                except KeyError:
                    print("Skipped: {}".format(list(trialDF.keys())[i]))
            conf = conf.copy(analysis.get_all_configs()[tunePath])

    model = modelPackage.Model(conf)
    model = model.to(device)

    if conf.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)
    elif conf.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)

    return model,optim

def summary(conf, model, dataloaders=None):
    if not dataloaders is None:
        batch = next(iter(dataloaders['test']))
        shape = batch['y'][0].shape
    else:
        shape = (conf.numT+1, conf.dimY)

    torchsummary.summary(model, shape)

def processData(conf):
    if conf.datasetType == 'classic':
        if not conf.customValidTest is None:
            retLoaders = {}
            retDatasets = {}
            for s in ['train', 'valid', 'test']:
                datasetType = conf.customValidTest[s].split(".")[-1]
                if datasetType == "npz":
                    currLoader,currDataset = loaders.npz(conf, customSet=s)
                elif datasetType == "hdf5":
                    currLoader,currDataset = loaders.hdf5(conf, customSet=s)
                else:
                    raise ValueError("dataset {} not valid.".format(conf.dataset))

                retLoaders[s] = currLoader
                retDatasets[s] = currDataset

            return retLoaders,retDatasets

        else:
            datasetType = conf.dataset.split(".")[-1]
            if datasetType == "npz":
                return loaders.npz(conf)
            elif datasetType == "hdf5":
                return loaders.hdf5(conf)
            else:
                raise ValueError("dataset {} not valid.".format(conf.dataset))

    elif conf.datasetType == 'multiCollapse':
        return loaders.multiCollapse(conf)

    else:
        raise ValueError("datasetType {} not valid.".format(conf.datasetType))

def predict(conf, model, dataloader, epoch, toSave=True, toReturn=False):
    device = next(model.parameters()).device
    model.eval()

    if toSave:
        if not os.path.exists(os.path.join(filesPath(conf), "predictions")):
            os.makedirs(os.path.join(filesPath(conf), "predictions"))

    predictions = {}
    for set in dataloader:
        with torch.no_grad():
            predictions[set] = defaultdict(list)

            for batchIndex, data in enumerate(dataloader[set]):
                true = {k: data[k].to(device) for k in data}

                pred = model(true)

                # predictions['true'].extend(list(true['y'].cpu().detach().numpy()))
                for k in true:
                    predictions[set][k].extend(list(true[k].cpu().detach().numpy()))
                predictions[set]['pred'].extend(list(pred['y'].cpu().detach().numpy()))

        for k in predictions[set]:
            predictions[set][k] = np.array(predictions[set][k])

        if conf.normalizeY:
            minMax = conf.multiCollapseMinMax
            if conf.fixedB:
                minMax = (minMax[0], minMax[1], minMax[3])
            for i in range(predictions[set]['pred'].shape[1]):
                predictions[set]['pred'][:,i] *= minMax[i][1] - minMax[i][0]
                predictions[set]['pred'][:,i] += minMax[i][0]


        if conf.filePredAppendix is None:
            filePred = os.path.join(filesPath(conf), "predictions", "{}-{}-{}.npz".format(".".join(conf.modelLoad.split('.')[:-1]), epoch, set))
        else:
            filePred = os.path.join(filesPath(conf), "predictions", "{}-{}-{}-{}.npz".format(".".join(conf.modelLoad.split('.')[:-1]), epoch, set, conf.filePredAppendix))

        if toSave:
            # np.savez_compressed(filePred, true=predictions['true'], pred=predictions['pred'])
            np.savez_compressed(filePred, **predictions[set])
            if not conf.has("nonVerbose") or conf.nonVerbose == False:
                print("Saved {}".format(filePred), flush=True)

    if toReturn:
        return predictions

def evaluate(conf, model, dataloader):
    device = next(model.parameters()).device
    
    lossFun = MyLoss(conf, device)
    slossFun = SLoss(conf, device)

        
    model.eval()
    with torch.no_grad():
        runningLoss = 0.
        runningMetrics = defaultdict(float)

        for batchIndex, data in enumerate(dataloader):
            true = {k: data[k].to(device) for k in data}

            pred = model(true)

            loss = lossFun(pred['y'], true['y'])

            runningLoss += loss.item()

            if conf.taskType == "classification":
                myPred = torch.softmax(pred['y'], dim=1)
                runningMetrics["acc"] += torch.sum(true['y'].reshape(true['y'].shape[0]) == myPred.max(dim=1).indices).float() / true['y'].shape[0]
            elif conf.taskType == "prediction":
                myPred = torch.exp(pred['y'])
                # runningMetrics["kld"] += sp.stats.entropy(true['y'].cpu(), myPred.cpu(), axis=2).mean(axis=1).mean(axis=0)
                runningMetrics["kld"] += sp.stats.entropy(true['y'].cpu(), myPred.cpu(), axis=2).sum(axis=1).mean(axis=0) #same as nn.KLDivLoss(reduction='batchmean')
            elif conf.taskType == "regressionL1":
                runningMetrics["mae"] += loss.item() #use directly loss
                runningMetrics["e"] += slossFun(pred['y'], true['y']).item() #use directly loss
            elif conf.taskType == "regressionL2":
                runningMetrics["mse"] += loss.item() #use directly loss
                runningMetrics["e"] += slossFun(pred['y'], true['y']).item() #use directly loss
            elif conf.taskType == "regressionMRE":
                runningMetrics["mre"] += loss.item() #use directly loss
                runningMetrics["e"] += slossFun(pred['y'], true['y']).item() #use directly loss
            elif conf.taskType == "regressionS":
                runningMetrics["e"] += loss.item() #use directly loss (is already sloss)
            else:
                raise ValueError("taskType {} not valid".format(conf.taskType))

        for k in runningMetrics:
            runningMetrics[k] /= len(dataloader)

        toReturn = [(runningLoss / len(dataloader)), runningMetrics]

        return toReturn

def runTrain(conf, model, optim, dataloaders, startEpoch, bestValidMetric=None):
    trainDataloader = dataloaders['train']
    validDataloader = dataloaders['valid']
    testDataloader = dataloaders['test']

    device = next(model.parameters()).device
    
    lossFun = MyLoss(conf, device)

    if conf.tensorBoard:
        writer = SummaryWriter(os.path.join(filesPath(conf),"tensorBoard"), flush_secs=10)#60)


    if not conf.earlyStopping is None:
        maxMetric = 0.
        currPatience = 0.

    if not os.path.exists(os.path.join(filesPath(conf),"models")):
        os.makedirs(os.path.join(filesPath(conf),"models"))

    if not conf.bestSign in ['<', '>']:
        raise ValueError("bestSign {} not valid".format(conf.bestSign))

    if bestValidMetric is None:
        if conf.bestSign == '>':
            bestValidMetric = - math.inf
        elif conf.bestSign == '<':
            bestValidMetric = math.inf

    globalBatchIndex = 0

    for epoch in range(startEpoch, startEpoch+conf.epochs):
        startTime = datetime.now()
        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            print("epoch {}".format(epoch), end='', flush=True)


        model.train()
        for batchIndex, data in enumerate(trainDataloader):
            true = {k: data[k].to(device) for k in data}

            model.zero_grad()

            pred = model(true)

            loss = lossFun(pred['y'], true['y'])
            loss.backward()

            optim.step()

            if conf.logEveryBatch:
                validLoss, validMetrics = evaluate(conf, model, validDataloader)
                testLoss, testMetrics = evaluate(conf, model, testDataloader)

                writerDictLoss = {
                    'train': loss.item(),
                    'valid': validLoss,
                    'test': testLoss,
                    }

                # writerDictMetrics = {}
                # for k in validMetrics: #same keys for train and valid
                #     writerDictMetrics[k] = {
                #         'valid': validMetrics[k],
                #         'test': testMetrics[k],
                #         }

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if conf.tensorBoard:
                        writer.add_scalars('batch-loss', writerDictLoss, globalBatchIndex)

                        # for k in writerDictMetrics:
                        #     writer.add_scalars('batch-{}_{}'.format(conf.trackMetric, k), writerDictMetrics[k], globalBatchIndex)
                        #     writer.add_scalars('batch-{}'.format(k), writerDictMetrics[k], globalBatchIndex)

                globalBatchIndex += 1



        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            print(": ", end='', flush=True)

        trainLoss, trainMetrics = evaluate(conf, model, trainDataloader)
        validLoss, validMetrics = evaluate(conf, model, validDataloader)
        testLoss, testMetrics = evaluate(conf, model, testDataloader)

        if conf.useTune:
            # tune.report(**{conf.bestKey: validMetrics[conf.bestKey].item()})
            # tune.report(**{conf.bestKey: validMetrics[conf.bestKey]})
            tuneDict = {}
            for setStr, lossValue, metricDict in [("train", trainLoss, trainMetrics), ("valid", validLoss, validMetrics), ("test", testLoss, testMetrics)]:
                tuneDict["/".join([setStr,"loss"])] = lossValue
                for k in metricDict:
                    tuneDict["/".join([setStr,k])] = metricDict[k]

            tune.report(**tuneDict)

        if not conf.bestKey in validMetrics.keys(): #conf.bestKey is used to control tune optimization
            raise ValueError("bestKey {} not present".format(conf.bestKey))

        writerDictLoss = {
            'train': trainLoss,
            'valid': validLoss,
            'test': testLoss,
            }

        writerDictMetrics = {}
        for k in trainMetrics: #same keys for train and valid
            writerDictMetrics[k] = {
                'train': trainMetrics[k],
                'valid': validMetrics[k],
                'test': testMetrics[k],
                }

        #save always last except when modelSave == "none"
        if conf.modelSave != "none":
            fileLast = os.path.join(filesPath(conf),"models","last.pt")

            if os.path.isfile(fileLast):
                os.remove(fileLast)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch,
                'bestValidMetric': bestValidMetric,
            }, fileLast)

        if conf.modelSave == "all":
            fileModel = os.path.join(filesPath(conf), "models", "epoch{}.pt")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch,
            'bestValidMetric': bestValidMetric,
            }, fileModel.format(epoch))
        elif conf.modelSave == "best":
            fileModel = os.path.join(filesPath(conf), "models", "best.pt")
            if not conf.bestKey in validMetrics.keys():
                raise ValueError("bestKey {} not present".format(conf.bestKey))
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] < bestValidMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] > bestValidMetric):
                bestValidMetric = validMetrics[conf.bestKey]
                
                if os.path.isfile(fileModel):
                    os.remove(fileModel)

                torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch,
                'bestValidMetric': bestValidMetric,
                }, fileModel)

        if conf.logCurves:
            currPath = os.path.join(filesPath(conf),"curves")
            Path(currPath).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(currPath,"epochs.dat"), 'a') as logFile:
                logFile.write("{}\n".format(epoch))

            for s in writerDictLoss:
                currPath = os.path.join(filesPath(conf),"curves","loss",s)
                Path(currPath).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(currPath,"points.dat"), 'a') as logFile:
                    logFile.write("{}\n".format(writerDictLoss[s]))

            for k in writerDictMetrics:
                for s in writerDictMetrics[k]:
                    currPath = os.path.join(filesPath(conf),"curves",k,s)
                    Path(currPath).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(currPath,"points.dat"), 'a') as logFile:
                        logFile.write("{}\n".format(writerDictMetrics[k][s]))


            os.path.join(filesPath(conf),"tensorBoard")

        if conf.tensorBoard:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                writer.add_scalars('loss', writerDictLoss, epoch)

                for k in writerDictMetrics:
                    # writer.add_scalars('{}_{}'.format(conf.trackMetric, k), writerDictMetrics[k], epoch)
                    writer.add_scalars('{}'.format(k), writerDictMetrics[k], epoch)

        endTime = datetime.now()
        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            # print("tr. loss {:0.3f}, tr. {} {:0.3f} - va. loss {:0.3f}, va. {} {:0.3f} - te. loss {:0.3f}, te. {} {:0.3f} ({}; exp. {})".format(trainLoss, conf.bestKey, trainMetrics[conf.bestKey], validLoss, conf.bestKey, validMetrics[conf.bestKey], testLoss, conf.bestKey, testMetrics[conf.bestKey], str(endTime-startTime).split('.', 2)[0], str((endTime-startTime)*(startEpoch+conf.epochs-1-epoch)).split('.', 2)[0]), flush=True)
            print("tr. {} {:0.3f} - va. {} {:0.3f} - te. {} {:0.3f} ({}; exp. {})".format(conf.bestKey, trainMetrics[conf.bestKey], conf.bestKey, validMetrics[conf.bestKey], conf.bestKey, testMetrics[conf.bestKey], str(endTime-startTime).split('.', 2)[0], str((endTime-startTime)*(startEpoch+conf.epochs-1-epoch)).split('.', 2)[0]), flush=True)

        if not conf.earlyStopping is None:
            if not conf.bestKey in validMetrics.keys():
                raise ValueError("bestKey {} not present".format(conf.bestKey))
            if not conf.bestSign in ['<', '>']:
                raise ValueError("bestSign {} not valid".format(conf.bestSign))
            #TODO: same of bestKey
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] > maxMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] < maxMetric):
                if currPatience >= conf.earlyStopping:
                    break
                currPatience += 1
            else:
                maxMetric = validMetrics[conf.bestKey]
                currPatience = 0
            
        #if not conf.earlyStopping is None:
        #    if validAcc < previousAccuracy:
        #        if currPatience >= conf.earlyStopping:
        #            break
        #        currPatience += 1
        #    else:
        #        currPatience = 0

        #if not conf.earlyStopping is None:
        #    if epoch >= conf.earlyStopping:
        #        if (validAcc <= previousAccuracies).all():
        #            break
        #    previousAccuracies[epoch%conf.earlyStopping] = validAcc

    time.sleep(30)#120) #time to write tensorboard


# def saveModel(conf, model, optim):
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optim_state_dict': optim.state_dict(),
#         }, os.path.join(filesPath(conf), "models", "model.pt"))

def loadModel(conf, device):
    fileToLoad = os.path.join(filesPath(conf), "models", conf.modelLoad)
    # if conf.modelSave == "best":
    #     fileToLoad = os.path.join(filesPath(conf), "models", "best.pt")
    # else:
    #     fileToLoad = os.path.join(filesPath(conf), "models", "epoch{}.pt".format(conf.loadEpoch))

    if not conf.has("nonVerbose") or conf.nonVerbose == False:
        print("Loading {}".format(fileToLoad), flush=True, end="")

    model, optim = makeModel(conf, device)

    checkpoint = torch.load(fileToLoad, map_location=torch.device('cpu'))#map_location=device)
    if not conf.has("nonVerbose") or conf.nonVerbose == False:
        print(" (epoch {})".format(checkpoint['epoch']), flush=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optim.load_state_dict(checkpoint['optim_state_dict'])
    return model,optim,checkpoint['epoch'],checkpoint['bestValidMetric']

def getLearningCurves(conf, metric='accuracy'):
    import tensorflow as tf
    
    sets = ['train', 'valid', 'test']

    path = {}
    for s in sets:
        path[s] = os.path.join(filesPath(conf), "tensorBoard", metric, s)

    logFiles = {}
    for s in sets:
        logFiles[s] = list(map(lambda f: os.path.join(path[s],f), sorted(os.listdir(path[s]))))

    values = {}
    for s in sets:
        values[s] = []
        for f in logFiles[s]:
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == 'loss' or v.tag == 'accuracy':
                        values[s].append(v.simple_value)

    return values

def getMaxValidEpoch(conf):
    values = getLearningCurves(conf)
    return np.argmax(values['valid'])

def getMaxValid(conf):
    try:
        values = getLearningCurves(conf)
        return values['valid'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def getMaxTest(conf):
    try:
        values = getLearningCurves(conf)
        return values['test'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def alreadyLaunched(conf):
    return os.path.isdir(os.path.join(filesPath(conf), "tensorBoard"))
