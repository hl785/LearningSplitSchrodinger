import numpy as np
from torch.utils import data
from scipy.sparse.linalg import expm
import copy

import model as md
import expsolve.fourier as fe           # Version 0.0.7

# ----------------------
# ----- DATAGEN.PY -----
# ----------------------

def gridGen(sampleRange, n):
    xRange = [-sampleRange, sampleRange]
    x = fe.grid1d(n, xRange)
    return x

def initCondGen(sampleRange, cent, x):
    xRange = [-sampleRange, sampleRange]
    u0 = lambda cent, x: np.exp(-(x-cent)**2/(2*0.25)).astype(np.complex128)
    # Eval
    u = u0(cent, x)
    # Norm
    u = u/fe.l2norm(u, xRange)
    return u

def wellGen(x):
    well = lambda x: x**4 - 10*x**2
    # Eval
    v = well(x)
    return v

def alphaBetaGen(splitLength, symSplit, splitType, randSmear = -1.0):
    # Add some more from here:
    # http://www.othmar-koch.org/splitting/
    if splitType in ['Trotter', 'trotter', 'sympEuler', 'Symp-Euler', 'symp-euler', 'SympEuler']:
        assert(not symSplit)
        assert(splitLength >= 1)
        gamma = np.zeros(2* splitLength)
        gamma[0:1] = 1.0
        gamma[splitLength:splitLength+1] = 1.0

    elif splitType in ['Strang', 'strang', 'leapfrog', 'leap', 'frog']:
        assert (splitLength >= 2)
        gamma = np.zeros(splitLength - 2)
        if not symSplit:
            alpha, beta = md.paramTransform(gamma)
            gamma = np.concatenate((alpha, beta), axis=None)

    elif splitType in ['Yoshida-4', 'Yoshida4', 'Yosh4', 'yosh4']:
        assert (splitLength >= 4)
        gamma = np.zeros(splitLength - 2)
        firstBetaInd = splitLength - splitLength//2 - 1
        omega = 1.0 / (2.0 - 2.0**(1.0/3.0))
        gamma[0:1] = [omega / 2.0]
        gamma[firstBetaInd:firstBetaInd+1] = [omega]
        if not symSplit:
            alpha, beta = md.paramTransform(gamma)
            gamma = np.concatenate((alpha, beta), axis=None)

    elif splitType in ['MultiStrang', 'Multistrang', 'Multileapfrog', 'Multileap', 'Multifrog', 'ms', 'MS', 'Ms', 'PolyStrang', 'polystrang', 'polyleapfrog', 'polyleap', 'polyfrog', 'ps', 'PS', 'Ps']:
        assert (splitLength >= 2)
        halfStep = 1/(2*(splitLength - 1))
        fullStep = 1/(splitLength - 1)
        gamma = np.zeros(splitLength - 2)
        firstBetaInd = splitLength - splitLength//2 - 1
        gamma[:] = fullStep
        gamma[0:1] = halfStep
        if not symSplit:
            alpha, beta = md.paramTransform(gamma)
            gamma = np.concatenate((alpha, beta), axis=None)

    elif splitType in ['Random', 'random', 'rand', 'rd']:
        assert (splitLength >= 2)
        gamma = np.random.uniform(-0.5/splitLength, 2.5/splitLength, splitLength - 2)   # Roughly 1/6 vals negative
        if not symSplit:
            alpha, beta = md.paramTransform(gamma)
            gamma = np.concatenate((alpha, beta), axis=None)

    else:
        assert False
    
    if randSmear > 0.0:
        offset = np.random.normal(0.0, randSmear, len(gamma))
        gamma += offset

    return gamma

class Dataset(data.IterableDataset):
    def __init__(self, T, delTFrac, sampleRange, spaDisc, subBatchSize):
        self.temporalChoices = np.array(np.meshgrid(T, delTFrac)).T.reshape(-1, 2)
        self.centChoicesParam = [-np.sqrt(5.0), 0.1]
        self.grid = gridGen(sampleRange, spaDisc)
        self.pot = wellGen(self.grid)
        self.sampleRange = sampleRange
        self.spaDisc = spaDisc
        self.subBatchSize = subBatchSize
        self.preCompVals = {}

        centMean = self.centChoicesParam[0]
        centSdDev = self.centChoicesParam[1]
        cent = np.random.normal(centMean, centSdDev)
        self.initData = [None, initCondGen(self.sampleRange, cent, self.grid), None, None, None]    # Only need a previous "final" value

    # Only need to compute matrix exp if temporal params/potential/etc change
    def fetchOrComp(self, tempParams):
        tempParamsTuple = (tempParams[0], tempParams[1])        # Dict wants a tuple as key, use ints to avoid problems with floating point =
        if tempParamsTuple not in self.preCompVals:             # If not stored then make
            timeRange = tempParams[0]
            timeDiscr = tempParams[0] * tempParams[1]
            
            xRange = fe.fixrange([-self.sampleRange, self.sampleRange], 1)
            timeStep = timeRange / timeDiscr
            potentialScal = -1j * timeStep * self.pot
            lapSym = fe.laplaciansymbol([len(self.pot)], xRange)
            lapSymScal = 1j * timeStep * np.array(lapSym)

            exactMat = expm(-1j*timeRange*(-fe.diffmatrix(2, self.spaDisc, xRange) + np.diag(self.pot)))
            self.preCompVals[tempParamsTuple] = [timeDiscr, potentialScal, lapSymScal, exactMat]  # store
        return self.preCompVals[tempParamsTuple]    # fetch

    def __iter__(self):
        current = self.initData
        stop = 10**6    # Set to -1 for inf data
        index = 0
        while True:
            components = [[] for _ in range(5)]
            for _ in range(self.subBatchSize):
                if index == stop:
                    return
                index += 1
                current = self.tranformData(current)

                for comp_idx in range(5):
                    components[comp_idx].append(current[comp_idx])
            stackComps = [np.stack(comp, axis=0) for comp in components]
            yield stackComps

    def tranformData(self, oldData):
        uInit = copy.copy(oldData[1])       # Avoid copy by ref, i.e. don't mutate old ref solution
        if np.random.rand() < 0.5:          # As linear equ can sum initial/final conditions (adds energy normally)
            centMean = self.centChoicesParam[0]
            centSdDev = self.centChoicesParam[1]
            cent = np.random.normal(centMean, centSdDev)
            uInit += initCondGen(self.sampleRange, cent, self.grid)

        if np.random.rand() < 0.5:          # Can rotate in complex plane
            rotFac = np.random.rand()
            uInit[:] *= np.cos(2.0*3.14159265359*rotFac) + 1j* np.sin(2.0*3.14159265359*rotFac)

        if np.random.rand() < 0.01:         # Occasionally reset to simple initial condition.
            centMean = self.centChoicesParam[0]
            centSdDev = self.centChoicesParam[1]
            cent = np.random.normal(centMean, centSdDev)
            uInit = initCondGen(self.sampleRange, cent, self.grid)
        
        xRange = fe.fixrange([-self.sampleRange, self.sampleRange], 1)
        uInit = uInit/fe.l2norm(uInit, xRange)      # Re-norm after potential add.

        index = np.random.choice(len(self.temporalChoices))
        tempParams = self.temporalChoices[index]    # Choose random tempParams
        [timeDiscr, potentialScal, lapSymScal, exactMat] = self.fetchOrComp(tempParams)
        uRefFinal = exactMat.dot(uInit)             # Calc ref sol (exact up to numerical precision)

        return [uInit, uRefFinal, timeDiscr, potentialScal, lapSymScal]

def postProcessLoss(diffs, losses, sampleRange):
    loss = np.average(np.array(losses))
    fnNormSqs = [fe.l2norm(np.array(diff), [-sampleRange, sampleRange])**2 for subdiffs in diffs for diff in subdiffs]  # Expand over 2 dims
    fnRMS = np.sqrt(np.average(fnNormSqs, axis=0))
    return loss, fnRMS

def postProcessGrads(diffs, losses, grads, sampleRange):
    loss, fnRMS = postProcessLoss(diffs, losses, sampleRange)
    grad = np.average(np.array(grads), axis=(0, 1))
    return loss, grad, fnRMS

def postProcessHess(hessians):
    hessian = np.average(np.array(hessians), axis=(0, 1))
    return hessian

def saveToWriter(loss, fnRMS, lossVal, fnRMSVal, symSplit, alphaBeta, writer, gen):
    writer.add_scalar('Loss/train', loss, gen)
    writer.add_scalar('L2 Error/train', fnRMS, gen)
    writer.add_scalar('Loss/validation', lossVal, gen)
    writer.add_scalar('L2 Error/validation', fnRMSVal, gen)

    if symSplit:
        alpha, beta = md.paramTransform(alphaBeta)
        gamma = np.concatenate((alpha, beta), axis=None)
        for j in range(len(gamma)):
            writer.add_scalar(f"Parameters/{j+1}", np.array(gamma[j]), gen)
        for j in range(len(alphaBeta)):
            writer.add_scalar(f"Reduced Parameters/{j+1}", np.array(alphaBeta[j]), gen)
    else: 
        for j in range(len(alphaBeta)):
            writer.add_scalar(f"Parameters/{j+1}", np.array(alphaBeta[j]), gen)