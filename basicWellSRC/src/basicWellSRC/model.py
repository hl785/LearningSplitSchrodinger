from jax.config import config
config.update("jax_enable_x64", True)

### -------------------
### ----- IMPORTS -----
### -------------------

import jax
from jax import numpy as jnp
from functools import partial

# --------------------
# ----- MODEL.PY -----
# --------------------

# ----- Helpers -----

# Go from reduced params to full params
def paramTransform(gamma):
    # TODO: include symmetric, consistent and unconstrained
    numParams = len(gamma)
    numBeta = numParams // 2
    numAlpha = numParams - numBeta

    alpha = gamma[:numAlpha]
    beta = gamma[numAlpha:]

    alphaSum = jnp.sum(alpha)
    betaSum = jnp.sum(beta)

    a = jnp.zeros(numParams + 2)
    b = jnp.zeros(numParams + 2)

    if numAlpha == numBeta + 1:
        a = a.at[:numAlpha].set(alpha)
        a = a.at[numAlpha].set(1 - 2 * alphaSum)
        a = a.at[(numAlpha + 1) :].set(jnp.flip(alpha, [0]))

        b = b.at[:numBeta].set(beta)
        b = b.at[numBeta].set(0.5 - betaSum)
        b = b.at[numBeta + 1].set(0.5 - betaSum)
        b = b.at[(numBeta + 2) : numParams + 1].set(jnp.flip(beta, [0]))

    elif numAlpha == numBeta:
        a = a.at[:numAlpha].set(alpha)
        a = a.at[numAlpha].set(0.5 - alphaSum)
        a = a.at[numAlpha + 1].set(0.5 - alphaSum)
        a = a.at[(numAlpha + 2) :].set(jnp.flip(alpha, [0]))

        b = b.at[:numBeta].set(beta)
        b = b.at[numBeta].set(1 - 2 * betaSum)
        b = b.at[(numBeta + 1) : numParams + 1].set(jnp.flip(beta, [0]))

    else:
        assert False

    return a, b

# Callback to file with access to jax
def numProc():
    return int(jax.device_count())

class ModelParams():
    def __init__(self, numLayers, symSplit, maxTimeDiscr):
        self.numLayers = numLayers
        self.symSplit = symSplit
        self.maxTimeDiscr = maxTimeDiscr

    def __hash__(self):
        return (
            hash(self.numLayers)
            ^ hash(self.symSplit)
            ^ hash(self.maxTimeDiscr)
        )

# ----- Forward/ODE solve -----

def potFlow(potentialScal, u, param):
        flow = jnp.exp(param * potentialScal)   # Vector exp = list of scalar exp => represents diag matrix exp
        return flow * u

def cfft(f):
    return jnp.fft.fftshift(jnp.fft.fft(f))

def cifft(f):
    return jnp.fft.ifft(jnp.fft.ifftshift(f))

def kinFlow(lapSymScal, u, param):
    esL = jnp.exp(param * lapSymScal)           # Vector exp = list of scalar exp => represents diag matrix exp
    return cifft(esL * cfft(u))

@partial(jax.jit, static_argnums=1)
def forward(gamma, modelParams, u, timeDiscr, potentialScal, lapSymScal):
    if modelParams.symSplit:
        alpha, beta = paramTransform(gamma)
    else:
        alpha = gamma[: modelParams.numLayers]
        beta = gamma[modelParams.numLayers :]

    assert len(alpha) == modelParams.numLayers
    assert len(beta) == modelParams.numLayers

    u = jnp.asarray(u, dtype=complex)                   # TODO: see why needed?

    def aFn(u):
        for i in range(modelParams.numLayers):
            u = potFlow(potentialScal, u, alpha[i])  # Alpha = first half of prams => represents potential opr
            u = kinFlow(lapSymScal, u, beta[i])      # Beta = second half of prams => represents kinetic opr
        return u

    def bFn(u):
        return u

    def condFn(i):
        return i < timeDiscr

    def loopBody(i, u):
        return jax.lax.cond(condFn(i), aFn, bFn, u)     # Hack to allow multiple timeDiscr, for loop over max only do stuff right num of time
    
    return jax.lax.fori_loop(0, modelParams.maxTimeDiscr, loopBody, u)

def forwardCast(gamma, modelParams, u, timeDiscr, potentialScal, lapSymScal):     # Protect if inputs are numpy array
    return forward(gamma, modelParams, jnp.array(u), int(timeDiscr), jnp.array(potentialScal), jnp.array(lapSymScal))

# ----- Backwards/loss -----

def lossFn(gamma, modelParams, u, uRefFinal, timeDiscr, potentialScal, lapSymScal):
    output = forward(gamma, modelParams, u, timeDiscr, potentialScal, lapSymScal)
    diff = output - uRefFinal
    return jnp.vdot(diff, diff).real, [output, diff]        # Loss is l2 vector norm squared

def lossFnNoAux(gamma, modelParams, u, uRefFinal, timeDiscr, potentialScal, lapSymScal):
    loss, _ = lossFn(gamma, modelParams, u, uRefFinal, timeDiscr, potentialScal, lapSymScal)
    return loss

@partial(jax.jit, static_argnums=1)
def jitLossVal(gamma, modelParams, u, uRefFinal, timeDiscr, potentialScal, lapSymScal):
    loss, [val, diff] = lossFn(gamma, modelParams, u, uRefFinal, timeDiscr, potentialScal, lapSymScal)
    return val, diff, loss

jitLossValBat = jax.pmap(jax.vmap(jitLossVal, in_axes=(None, None, 0, 0, 0, 0, 0)), in_axes=(None, None, 0, 0, 0, 0, 0), static_broadcasted_argnums=1)

def jitLossValBatCast(alphaBeta, modelParams, uInits, uRefFinals, timeDiscrs, potentialScals, lapSymScals):  # Protect if inputs are numpy array
    return jitLossValBat(alphaBeta, modelParams, jnp.array(uInits), jnp.array(uRefFinals), jnp.array(timeDiscrs), jnp.array(potentialScals), jnp.array(lapSymScals))

@partial(jax.jit, static_argnums=1)
def jitLossGradVal(alphaBeta, modelParams, uInit, uRefFinal, timeDiscr, potentialScal, lapSymScal):
    # print("Tracing")    # Uncomment to check if being recompiled
    [loss, [val, diff]], grad = jax.value_and_grad(lossFn, has_aux=True)(alphaBeta, modelParams, uInit, uRefFinal, timeDiscr, potentialScal, lapSymScal)
    return val, diff, loss, grad

# Use both pmap and vmap for max speed => implies "batches" are over 2 dimensions
lossGradValBat = jax.pmap(jax.vmap(jitLossGradVal, in_axes=(None, None, 0, 0, 0, 0, 0)), in_axes=(None, None, 0, 0, 0, 0, 0), static_broadcasted_argnums=1)

def lossGradValBatCast(alphaBeta, modelParams, uInits, uRefFinals, timeDiscrs, potentialScals, lapSymScals):  # Protect if inputs are numpy array
    return lossGradValBat(alphaBeta, modelParams, jnp.array(uInits), jnp.array(uRefFinals), jnp.array(timeDiscrs), jnp.array(potentialScals), jnp.array(lapSymScals))

@partial(jax.jit, static_argnums=1)
def jitHessian(alphaBeta, modelParams, uInit, uRefFinal, timeDiscr, potentialScal, lapSymScal):
    # print("Tracing")    # Uncomment to check if being recompiled
    hessian = jax.hessian(lossFnNoAux)(alphaBeta, modelParams, uInit, uRefFinal, timeDiscr, potentialScal, lapSymScal)
    return hessian

# Use both pmap and vmap for max speed => implies "batches" are over 2 dimensions
hessianBat = jax.pmap(jax.vmap(jitHessian, in_axes=(None, None, 0, 0, 0, 0, 0)), in_axes=(None, None, 0, 0, 0, 0, 0), static_broadcasted_argnums=1)

def hessianBatCast(alphaBeta, modelParams, uInits, uRefFinals, timeDiscrs, potentialScals, lapSymScals):  # Protect if inputs are numpy array
    return hessianBat(alphaBeta, modelParams, jnp.array(uInits), jnp.array(uRefFinals), jnp.array(timeDiscrs), jnp.array(potentialScals), jnp.array(lapSymScals))