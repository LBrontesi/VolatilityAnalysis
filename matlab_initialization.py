"""arch volatility processes with the assignment's MATLAB presample defaults.

MathWorks estimate/infer use the mean squared offset-adjusted responses for V0,
positive sqrt(V0) innovations for ARCH/GARCH/GJR, and zero innovations for
EGARCH. arch's weighted backcast and symmetric presample shock differ.
These subclasses change initialization only; arch handles likelihood and fitting.
"""
import numpy as np
from arch.univariate import GARCH, EGARCH


class AssignmentGARCH(GARCH):
    def backcast(self, resids):
        return float(np.mean(np.asarray(resids)**2))

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        backcast = self.backcast(resids)
        if self.o:
            if (self.p, self.o, self.q) != (1, 1, 1):
                raise ValueError("Assignment GJR initialization supports (1,1,1)")
            # arch uses gamma*V0/2 for its missing asymmetric shock; MATLAB's
            # positive E0 contributes zero. A modified common backcast makes
            # the first recursion omega+(alpha+beta)*V0 in both cases.
            _, alpha, gamma, beta = parameters
            denominator = alpha + gamma/2 + beta
            if denominator != 0:
                backcast *= (alpha+beta)/denominator
        return super().compute_variance(parameters, resids, sigma2, backcast, var_bounds)


class AssignmentEGARCH(EGARCH):
    def backcast(self, resids):
        return float(np.log(np.mean(np.asarray(resids)**2)))

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        if (self.p, self.o, self.q) != (1, 1, 1):
            raise ValueError("Assignment EGARCH initialization supports (1,1,1)")
        _, alpha, _, beta = parameters
        backcast = self.backcast(resids)
        if beta != 0:
            # arch substitutes E|z| for absent |z0|. MATLAB uses z0=0,
            # giving omega+beta*log(V0)-alpha*sqrt(2/pi) at t=0.
            adjusted = backcast-alpha*np.sqrt(2/np.pi)/beta
            return super().compute_variance(parameters, resids, sigma2, adjusted, var_bounds)
        # With beta=0 a synthetic zero presample shock initializes the first
        # observed variance without needing division by beta.
        padded = np.concatenate(([0.], resids))
        variances = np.empty(len(padded))
        bounds = np.vstack((var_bounds[:1], var_bounds))
        super().compute_variance(parameters, padded, variances, backcast, bounds)
        sigma2[:] = variances[1:]
        return sigma2
