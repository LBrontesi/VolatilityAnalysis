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


def dcc_objective_gradient(residuals, qbar, psi):
    """DCC objective and analytic gradient for stable BFGS estimation."""
    z = np.asarray(residuals, dtype=float)
    qbar = np.asarray(qbar, dtype=float)
    a, fraction = expit(psi)
    b = (1-a)*fraction
    da = np.array([a*(1-a), 0.])
    db = np.array([-a*b, b*(1-fraction)])
    q = qbar.copy()
    dq = np.zeros((2, *q.shape))
    total, gradient = 0., np.zeros(2)
    for i, observation in enumerate(z):
        if i:
            outer = np.outer(z[i-1], z[i-1])
            dq = (-(da+db)[:, None, None]*qbar + da[:, None, None]*outer
                  + db[:, None, None]*q + b*dq)
            q = (1-a-b)*qbar + a*outer + b*q
        diagonal = np.diag(q)
        scale = np.sqrt(np.outer(diagonal, diagonal))
        p = q/scale
        dp_diagonal = np.diagonal(dq, axis1=1, axis2=2)/diagonal
        dp = dq/scale - .5*p*(dp_diagonal[:, :, None]+dp_diagonal[:, None, :])
        sign, logdet = np.linalg.slogdet(p)
        if sign <= 0:
            return np.inf, np.zeros(2)
        inverse = np.linalg.solve(p, np.eye(p.shape[0]))
        solved = inverse@observation
        total += .5*(logdet+observation@solved)
        gradient += .5*np.einsum('ij,kji->k', inverse-np.outer(solved, solved), dp)
    return float(total), gradient
