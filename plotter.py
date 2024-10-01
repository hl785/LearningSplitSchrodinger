import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os.path

import model as md
import expsolve.fourier as fe

# ----------------------
# ----- PLOTTER.PY -----
# ----------------------

def error(uMl, uRefSubset, xRange):
    return fe.l2norm(uRefSubset - uMl, xRange)**2

def plotFn(l2SquLoss, L2ValLoss, initParams, params, save = True, path = 'fig.svg'):
    plt.clf()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('l2SquLoss')
    ax2.set_ylabel('L2ValLoss')
    ax3.set_ylabel('AlphaBeta')

    itr = range(len(l2SquLoss))
    handle1 = ax1.plot(itr, l2SquLoss, color = plt.cm.viridis(0.0), label = 'l2SquLoss')
    # ax1.annotate(f'({itr[-1]}, {l2SquLoss[-1]:-2.3})', xy=(itr[-1], l2SquLoss[-1]), textcoords='data')
    itr = range(len(L2ValLoss))
    handle2 = ax2.plot(itr, L2ValLoss, color = plt.cm.viridis(0.5), label = 'L2ValLoss')
    ax2.annotate(f'({itr[-1]}, {L2ValLoss[-1]:-2.3})', xy=(itr[-1], L2ValLoss[-1]), textcoords='data')

    color = ['b', 'r']*(len(params)//2)
    newParams = copy.copy(params)
    newParams[::2] = params[:(len(params)//2)]
    newParams[1::2] = params[(len(params)//2):]
    ax3.bar(fe.grid1d(len(params), xrange=[0, itr[-1]]), newParams, itr[-1] / len(params), alpha=0.2, color = color, linewidth = 0.1)
    newParams = copy.copy(initParams)
    newParams[::2] = initParams[:(len(initParams)//2)]
    newParams[1::2] = initParams[(len(initParams)//2):]
    ax3.bar(fe.grid1d(len(params), xrange=[0, itr[-1]]), newParams, itr[-1] / len(params), edgecolor='black', color='none', linewidth = 0.1)

    handle3 = mpatches.Patch(color='blue', alpha=0.2, label='Alphas')
    handle4 = mpatches.Patch(color='red', alpha=0.2, label='Betas')
    handle5 = mpatches.Patch(facecolor='none', edgecolor='black', label='InitABs')
    ax1.legend(handles = handle1+handle2+[handle3]+[handle4]+[handle5], loc='best')
    
    ax2.spines['left'].set_position(('outward', 60))
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')

    ax1.yaxis.label.set_color(plt.cm.viridis(0.0))
    ax2.yaxis.label.set_color(plt.cm.viridis(0.5))
    ax1.tick_params(axis='y', colors=plt.cm.viridis(0.0))
    ax2.tick_params(axis='y', colors=plt.cm.viridis(0.5))

    if save:
        plt.tight_layout()
        plt.savefig(path)
    else:
        plt.ion()
        plt.show()
        plt.pause(0.0001)

def saveParams(path, symSplit, alphaBeta):
    gamma = alphaBeta
    if symSplit:
        alpha, beta = md.paramTransform(alphaBeta)
        gamma = np.concatenate((alpha, beta), axis=None)

    try:
        os.makedirs(fileName, exist_ok=True)
    except Exception:
        pass

    fileName = os.path.join(path, 'params.json')
    with open(fileName, 'w', encoding='utf-8') as f:
        data = {}
        data['Parameters'] = np.array(alphaBeta).tolist()
        data['Transformed Parameters'] = np.array(gamma).tolist()
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()