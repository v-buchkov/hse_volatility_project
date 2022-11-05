"""Script for """
import numpy as np
import scipy.optimize
from typing import List


class OLSEstimator:
    """
    A class for predicting difference in variances via 1-lag autoregressive model estimated by OLS.
    As model is 1-lag, OLS provides unbiased and consistent estimators (no multicollinearity, unlike 2+ lags models).

    ...

    Attributes
    ----------
    var_onshore : list
        list of variance of onshore asset (volatility squared)
    var_offshore : list
        list of variance of offshore asset (volatility squared)
    days : float
        length of window, for which we try to predict the volatility features
    n_sample : int
        number of variance points in the sample
    var_diff : list
        list of realized variance differences
    coefficient : list
        list of OLS-fitted coefficients

    Methods
    -------
    variance_difference_list():
        Returns list of realized variance differences.
    predicted_list(constants):
        Generates list of predicted variance differences from given constants.
    mse_function(constants):
        Calculates Mean Squared Error via calling predicted_list(constants).
    minimize_mse():
        Minimizes Mean Squared Error via iterating over potential solutions.
    """
    def __init__(self, vol_onshore: List[float], vol_offshore: List[float], days_strategy: int):
        self.var_onshore = [sigma ** 2 for sigma in vol_onshore]
        self.var_offshore = [sigma ** 2 for sigma in vol_offshore]
        self.days = days_strategy

        self.n_sample = min(len(self.var_onshore), len(self.var_offshore))
        self.var_diff = self.variance_difference_list()
        self.coefficient = self.minimize_mse()

    def variance_difference_list(self):
        """
        Returns list of realized variance differences.

        Returns
        -------
        var_diff : list
            List of realized variance differences.
        """
        return [self.var_onshore[i] - self.var_offshore[i] for i in range(self.n_sample)]

    def predicted_list(self, constants: List[float]):
        """
        Generates list of predicted variance differences from given constants.
        Formula used: diff[i + 1] = constants[0] + constants[1] * diff[i - 1].

        Returns
        -------
        var_predicted : list
            List of predicted variance differences.
        """
        alpha = constants[0]
        beta = constants[1]

        var_predicted = np.zeros(self.n_sample)

        for i in range(len(var_predicted)):
            if i == 0:
                var_predicted[i] = alpha / (1 - beta)
            else:
                var_predicted[i] = alpha + beta * var_predicted[i - 1]

        return var_predicted

    def mse_function(self, constants: List[float]):
        """
        Calculates Mean Squared Error via calling predicted_list(constants).

        Returns
        -------
        mse : float
            Mean Squared Error for given constants.
        """
        predicted = self.predicted_list(constants)
        return sum([(self.var_diff[i + self.days] - predicted[i]) ** 2 for i in range(len(self.var_diff) - self.days)])

    def minimize_mse(self):
        """
        Minimizes Mean Squared Error via iterating over potential solutions.
        Uses scipy optimization for unconstrained minimization.

        Returns
        -------
        optimal_constants : list
            List of optimal (MSE-minimizing) constants.
        """
        constants = np.array([0.001, 0.001])

        optimal_constants = scipy.optimize.minimize(self.mse_function, constants).x

        return optimal_constants


"""To be developed further"""
# class MLEstimator:
#
#     def __init__(self, vol_onshore: List[float], vol_offshore: List[float]):
#         self.var_onshore = [sigma ** 2 for sigma in vol_onshore]
#         self.var_offshore = [sigma ** 2 for sigma in vol_offshore]
#         self.coefficient = self.optimize_loglikelihood()
#
#     def variance_differnce_list(self, constants):
#
#         omega = constants[0]
#         alpha = constants[0]
#
#         length = min(len(self.var_onshore), len(self.var_offshore))
#
#         # Initializing an empty array
#         variance_diff = np.zeros(length)
#
#         # Filling the array, if i == 0 then uses the average realized variance
#         for i in range(length):
#             if i == 0:
#                 variance_diff[i] = omega / (1 - alpha)
#             else:
#                 variance_diff[i] = omega + alpha * variance_diff[i - 1]
#
#         return variance_diff
#
#     def loglikelihood(self, constants):
#
#         variance_diff = self.variance_differnce_list(constants)
#         returns = self.returns
#         sample_size = len(variance_diff)
#
#         # print(constants, [round(100 * var, 4) for var in variance])
#
#         # print([np.log(var) for var in variance[1:]])
#
#         loglikelihood =
#
#         return loglikelihood
#
#     def optimize_loglikelihood(self):
#
#         constants = np.array([0.001])
#
#         optimal_constants = scipy.optimize.minimize(self.loglikelihood, constants,
#                                                     bounds=[(0.0001, None), (0.0001, None), (0.0001, None)])
#
#         return optimal_constants.x
