"""Engle DCC likelihood, retaining the MATLAB helper's return convention."""
import numpy as np
from scipy.special import expit


def dcc_correlations(residuals, qbar, a, b):
    z = np.asarray(residuals, dtype=float)
    q = np.asarray(qbar, dtype=float).copy()
    correlations = np.empty((len(z), z.shape[1], z.shape[1]))
    for i in range(len(z)):
        if i:
            q = (1-a-b)*qbar + a*np.outer(z[i-1], z[i-1]) + b*q
        scale = np.sqrt(np.diag(q))
        correlations[i] = q / np.outer(scale, scale)
    return correlations


def fDCC_LogLikelihood(my_star, mQbar, vPsi):
    z = np.asarray(my_star, dtype=float)
    qbar = np.asarray(mQbar, dtype=float)
    if z.ndim != 2 or qbar.shape != (z.shape[1], z.shape[1]):
        raise ValueError("Expected residuals (time, assets) and matching Qbar")
    if not np.isfinite(z).all() or not np.isfinite(qbar).all():
        raise ValueError("Inputs must be finite")
    a = float(expit(vPsi[0]))
    b = float((1-a)*expit(vPsi[1]))
    total = 0.0
    try:
        for observation, correlation in zip(z, dcc_correlations(z, qbar, a, b)):
            sign, logdet = np.linalg.slogdet(correlation)
            if sign <= 0:
                return np.inf, a, b
            total += .5*(logdet + observation @ np.linalg.solve(correlation, observation))
    except np.linalg.LinAlgError:
        total = np.inf
    return float(total), a, b
