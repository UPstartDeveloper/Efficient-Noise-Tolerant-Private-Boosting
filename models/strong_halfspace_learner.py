import warnings

import numpy as np
from scipy.special import xlogy
from sklearn import ensemble
# from sklearn.base import is_regressor
# from sklearn.utils import check_random_state
# from sklearn.utils.validation import _check_sample_weight

# from diffprivlib.accountant import BudgetAccountant
# from diffprivlib.mechanisms import Gaussian
# from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args, check_random_state
# from diffprivlib.validation import DiffprivlibMixin


class DPAdaBoostClassifier(ensemble.AdaBoostClassifier):
    r"""The 'Strong Halfspace Learner' (aka HS-StL) classifier by Bun et. al (2020).

    TODO

    Attributes
    ----------
    TODO

    Examples
    --------
    >>> TODO

    See also
    --------
        TODO: point to the Gaussian mechanism, and the original AdaBoostClassifier

    References
    ----------
    .. [TODO]
    """

    # _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
    #     ensemble.AdaBoostClassifier,
    #     "estimator", "n_estimators",
    #     "learning_rate", "algorithm",
    #     "random_state")

    def __init__(self, estimator,
                 tau,  # e.g., the difference between the cov of the classes
                 k,  # e.g., I guess this'd be the avg of the cov of the classes
                 random_state=None,
                 alpha=.95, beta=0.05, epsilon=0.1, delta=0.01,
                 *args, **unused_args):
        
        # A: for docs purposes - these are what all the vars mean
        desired_final_accuracy = self.alpha = alpha
        probability_of_learning_failure = self.beta = beta
        desired_final_privacy = self.epsilon = epsilon
        desired_final_privacy_approx = self.delta = delta
        density_of_lazybregboost = self.k = k
        margin_of_halfspace = self.tau = tau
        # parameters for the mechanism - will be set to Gaussian as default for now
        self.epsilon = epsilon
        self.delta = delta

        # B: configure more params of the boosting algorithm
        self.sigma = ... # TODO[Zain]
        self.T = n_estimators = ...  # TODO[Zain]

        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         algorithm='SAMME.R', random_state=random_state,
                         *args, **unused_args)


    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using a DP version of the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            loss = 1 - 0.5 * (np.linalg.norm(y - y_predict_proba, 1))
            # ✅ Debug[Zain]: double check this against what's in the paper 
            sample_weight *= np.exp(-1 * self.learning_rate * np.sum(loss))
            sample_weight_all_prod = np.prod(sample_weight)
            sample_weight = np.repeat(sample_weight_all_prod, X.shape[0])

        # Change[Zain] - go from returning 1, to the actual weight
        # return sample_weight, 1.0, estimator_error
        return sample_weight, estimator_weight, estimator_error
    
    def fit(self, X, y):
        # as per the paper, we initialize the sample_weights to uniform vector
        # (whereas the default would be to use all 1's)
        sample_weight = np.repeat(self.k, X.shape[0])
        return super().fit(X, y, sample_weight)