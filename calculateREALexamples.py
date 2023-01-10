#!/usr/bin/env python

import numpy as np

printLatex = True

loadFile = "files/configWinCNN2W3/predictions/best-33-all-R3.1.npz"

if __name__ == '__main__':
    predictions = np.load(loadFile)

    if 'true' in predictions.files:
        trueN = predictions['true']
    elif 'y' in predictions.files:
        trueN = predictions['y']

    predN = predictions['pred']

    true = np.zeros_like(trueN)
    pred = np.zeros_like(predN)

    if predN.shape[1] == 4:
        parIndex = ["$y_0$", "$a$", "$B$", "$w_1$"]
        minMax = (  #from data creation
                (0.0005, 0.01),
                (0.1, 1),
                (50, 1000),
                (0.001, 0.01),
            )

    elif predN.shape[1] == 3:
        parIndex = ["$y_0$", "$a$", "$w_1$"]
        minMax = (  #from data creation
                (0.0005, 0.01),
                (0.1, 1),
                (0.001, 0.01),
            )
    else:
        raise ValueError("Something wrong")

    mseN = []
    maeN = []
    mreN = []
    mse = []
    mae = []
    mre = []
    for p in range(predN.shape[1]):
        if np.max(trueN) > 2: # not normalized y
            trueN[:,p] -= minMax[p][0]
            trueN[:,p] /= minMax[p][1] - minMax[p][0]
            predN[:,p] -= minMax[p][0]
            predN[:,p] /= minMax[p][1] - minMax[p][0]

        mseN.append(np.mean((trueN[:, p] - predN[:, p]) ** 2))
        maeN.append(np.mean(np.abs(trueN[:, p] - predN[:, p])))
        ## mre = np.mean(np.abs(true[:, p] - pred[:, p]) / (true[:, p]+10**(-5)))
        # mre = []
        # for i in range(len(true)):
        #     if true[i,p] != 0:
        #         mre.append(np.abs(true[i, p] - pred[i, p]) / true[i, p])
        # mre = np.array(mre)
        # mre = np.mean(mre)

        # mreN.append(np.mean(np.abs(true[:, p] - pred[:, p]) / (pred[:, p])))
        # mreN.append(np.mean(np.abs(true[:, p] - pred[:, p]) / (true[:, p])))

        # print("{} (normalized) - {:.2g} - {:.2g} - {:.2g}".format(parIndex[p], mse, mae, mre))

        true[:,p] = trueN[:,p] * (minMax[p][1] - minMax[p][0])
        true[:,p] += minMax[p][0]

        pred[:,p] = predN[:,p] * (minMax[p][1] - minMax[p][0])
        pred[:,p] += minMax[p][0]

        mse.append(np.mean((true[:, p] - pred[:, p]) ** 2))
        mae.append(np.mean(np.abs(true[:, p] - pred[:, p])))
        # mre.append(np.mean(np.abs(true[:, p] - pred[:, p]) / pred[:, p]))
        mre.append(np.mean(np.abs(true[:, p] - pred[:, p]) / true[:, p]))
        # print("{} - {:.9f} - {:.2g} - {:.2g}".format(parIndex[p], mse, mae, mre))

    print("======================================================================  MSE etc.")
    print("\\begin{tabular}{c|cccc}")
    print("\tParameter & MSE (nor.) & MAE (nor.) & MAE & MRE \\\\")
    print("\t\\hline")
    for p in range(pred.shape[1]):
        print("\t{} & {:.2g} & {:.2g} & {:.2g} & {:.2g} \\\\".format(parIndex[p], mseN[p], maeN[p], mae[p], mre[p]))
    print("\\end{tabular}")

    print("======================================================================  Examples")
    print("\\begin{{tabular}}{{{}|{}}}".format("c"*pred.shape[1], "c"*pred.shape[1]))
    print("\t\multicolumn{{{}}}{{c}}{{True}} & \multicolumn{{{}}}{{c}}{{Predicted}} \\\\".format(pred.shape[1], pred.shape[1]))
    print("\t" + " & ".join(parIndex + parIndex) + "\\\\")
    print("\t\\hline")
    for i in range(10):
        toPrint = ["{:.4f}".format(n) for n in true[i]] + ["{:.4f}".format(n) for n in pred[i]]
        print("\t" + " & ".join(toPrint)+" \\\\")
    print("\\end{tabular}")

    print("======================================================================  Examples MAE (MAEN%, MRE%)")
    print("\\begin{{tabular}}{{{}}}".format("c"*pred.shape[1]))
    print("\t" + " & ".join(parIndex) + "\\\\")
    print("\t\\hline")

    for i in range(10):
        # toPrint = ["${:.1f}\\%$".format(np.abs(p-t)/t*100) for t,p in zip(true[i], pred[i])]
        # toPrint = ["${:.1f}\\%$".format(np.abs(p-t)/p*100) for t,p in zip(true[i], pred[i])]
        toPrint = ["${:.2g}$ (${:.1f}\\%$, ${:.1f}\\%$)".format(np.abs(p-t), np.abs(pn-tn)*100, np.abs(p-t)/t*100) for t,p,tn,pn in zip(true[i], pred[i], trueN[i], predN[i])]
        print("\t" + " & ".join(toPrint)+" \\\\")
    print("\\end{tabular}")
