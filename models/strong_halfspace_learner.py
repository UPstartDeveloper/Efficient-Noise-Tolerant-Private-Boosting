import warnings

import numpy as np
from sklearn import ensemble
from sklearn.base import is_regressor
from sklearn.utils.validation import _check_sample_weight

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

    def __init__(self, estimator, n_estimators=50,
                 algorithm='SAMME.R', random_state=None,
                 alpha=.95, beta=0.05, epsilon=0.1, delta=0.01,
                 tau=..., k=...,
                 *args, **unused_args):
        
        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         algorithm=algorithm, random_state=random_state, *args)
        
        desired_final_accuracy = alpha
        probability_of_learning_failure = beta
        desired_final_privacy = epsilon
        desired_final_privacy_approx = delta
        density_of_lazybregboost = k
        margin_of_halfspace = tau

        # self.accountant = BudgetAccountant.load_default(accountant)
        # parameters for the mechanism - will be set to Gaussian as default for now
        self.epsilon = epsilon
        self.delta = delta
        # self.data_norm = data_norm  # TODO[Zain]: delete?
        # self.classes_ = None

        # warn_unused_args(unused_args)

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : class

        """
        self._validate_params()

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, np.float64, copy=True, only_non_negative=True
        )
        sample_weight /= sample_weight.sum()

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps

        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            # do not clip sample weights that were exactly zero originally
            sample_weight[zero_weight_mask] = 0.0

            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self
