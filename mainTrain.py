from jax.config import config
config.update("jax_enable_x64", True)           # AutoGrad needs more accuracy

### -------------------
### ----- IMPORTS -----
### -------------------

import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import time
import optax

import model as md
import dataGen as dg
import plotter as pt

### ----------------------
### ----- Parameters -----
### ----------------------

# Temporal choices                              # Assume increasing => T[-1] = max(T[i]), unitNumDelT[-1] = max(unitNumDelT[i])
T = np.array([10])                              # Total length of time horizon
unitNumDelT = np.array([7])                    # Number of sub time steps per increase of T by 1

# Spatial choices
sampleRange = 10                                # Model PDE on [-sampleRange, sampleRange] 
spaDisc = 200                                   # Fineness of spatial discretisation/number of Fourier modes 

# Splitting choices
splitLength = 5                                # Number of alpha_i's/beta_i's
symSplit = True                                 # Force learned params to be symmetric
splitType = 'Random'                            # Initial conditions for learning
randSmear = 0.01                                # Slightly perturb from initCond (to avoid local min)

# Machine Learning Hyperparameters
subBatchSize = 200                              # Size of batch per processing unit => total batch size = subBatchSize * numProc
learnRateInit = 0.01                             # TODO: Explain

# Consequences of choices
# TODO: calculate max{T[i] * unitNumDelT[j]} over i, j
modelParams = md.ModelParams(splitLength, symSplit, int(T[-1] * unitNumDelT[-1]))
alphaBetaInit = dg.alphaBetaGen(splitLength, symSplit, splitType, randSmear)
alphaBeta = alphaBetaInit                       # Keep copy for plotting at end
batchSize = md.numProc()                        # Use all possible devices TODO: maybe make this max? 

### ------------------
### ----- Set Up -----
### ------------------

seed = np.random.get_state()
np.random.seed(0)                               # Force validation set equal (to enable cross run comparison)
dataset = dg.Dataset(T, unitNumDelT, sampleRange, spaDisc, subBatchSize)
dataloader = data.DataLoader(dataset, batch_size=batchSize)
uInitsVal, uRefFinalsVal, timeDiscrsVal, potentialScalsVal, lapSymScalsVal = dataloader.__iter__().__next__()
np.random.set_state(seed)

lrnRateFn = optax.constant_schedule(learnRateInit)
optimizer = optax.adam(lrnRateFn)
optState = optimizer.init(alphaBeta)

tic = time.perf_counter()
valsVal, diffsVal, lossesVal, gradsVal = md.lossGradValBatCast(alphaBeta, modelParams, uInitsVal, uRefFinalsVal, timeDiscrsVal, potentialScalsVal, lapSymScalsVal)
toc = time.perf_counter()

writer = SummaryWriter('.')
lossVal, gradVal, fnRMSVal = dg.postProcessGrads(diffsVal, lossesVal, gradsVal, sampleRange)
dg.saveToWriter(lossVal, fnRMSVal, lossVal, fnRMSVal, symSplit, alphaBeta, writer, 0)

### -----------------
### ----- TRAIN -----
### -----------------

l2SquLoss = [lossVal]
L2ValLoss = [fnRMSVal]

for i, [uInits, uRefFinals, timeDiscrs, potentialScals, lapSymScals] in enumerate(dataloader):
    # Training
    vals, diffs, losses, grads = md.lossGradValBatCast(alphaBeta, modelParams, uInits, uRefFinals, timeDiscrs, potentialScals, lapSymScals)
    loss, grad, fnRMS = dg.postProcessGrads(diffs, losses, grads, sampleRange)

    # Validation
    valsVal, diffsVal, lossesVal, gradsVal = md.lossGradValBatCast(alphaBeta, modelParams, uInitsVal, uRefFinalsVal, timeDiscrsVal, potentialScalsVal, lapSymScalsVal)
    lossVal, gradVal, fnRMSVal = dg.postProcessGrads(diffsVal, lossesVal, gradsVal, sampleRange)

    dg.saveToWriter(loss, fnRMS, lossVal, fnRMSVal, symSplit, alphaBeta, writer, i+1)
    l2SquLoss.append(lossVal)
    L2ValLoss.append(fnRMSVal)

    if (i % 5 == 0):   
        pt.saveParams("", symSplit, alphaBeta)

    updates, optState = optimizer.update(grad, optState, alphaBeta)
    alphaBeta = optax.apply_updates(alphaBeta, updates)

### -------------------
### ----- Wrap Up -----
### -------------------

pt.saveParams("", symSplit, alphaBeta)
writer.close()

# Save change in params
gammaInit = alphaBetaInit
gamma = alphaBeta
if symSplit:
    alpha, beta = md.paramTransform(alphaBetaInit)
    gammaInit = np.concatenate((alpha, beta), axis=None)
    alpha, beta = md.paramTransform(alphaBeta)
    gamma = np.concatenate((alpha, beta), axis=None)
    
fileName = 'ParamLossEvolution.svg'
pt.plotFn(l2SquLoss, L2ValLoss, gammaInit, gamma, True, fileName)