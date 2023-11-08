import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_consistent_length,
)


class ExtensionDummyRegressor(DummyRegressor):
    """Extension regressor that makes predictions using simple rules.

    A new strategy "fracsum" has been added, which returns the sum of
    fractional parts in the training set similarly to "mean" and
    "median" strategies.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant", "fracsum"},
        default="mean". Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.
        * "fracsum": always predicts the sum of fractional parts in the
        training set.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int
        Number of outputs.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        y = check_array(y, ensure_2d=False, input_name="y")
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y, sample_weight)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if self.strategy == "mean":
            self.constant_ = np.average(y, axis=0, weights=sample_weight)

        elif self.strategy == "median":
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [
                    _weighted_percentile(
                        y[:, k], sample_weight, percentile=50.0
                    )
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "quantile":
            if self.quantile is None:
                raise ValueError(
                    "When using `strategy='quantile', you have to specify the "
                    "desired quantile in the range [0, 1]."
                )
            percentile = self.quantile * 100.0
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [
                    _weighted_percentile(
                        y[:, k], sample_weight, percentile=percentile
                    )
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "constant":
            if self.constant is None:
                raise TypeError(
                    "Constant target value has to be specified "
                    "when the constant strategy is used."
                )

            self.constant_ = check_array(
                self.constant,
                accept_sparse=["csr", "csc", "coo"],
                ensure_2d=False,
                ensure_min_samples=0,
            )

            if self.n_outputs_ != 1 and self.constant_.shape[0] != y.shape[1]:
                raise ValueError(
                    "Constant target value should have shape (%d, 1)."
                    % y.shape[1]
                )

        elif self.strategy == "fracsum":
            self.constant_ = np.sum(y % 1)

        self.constant_ = np.reshape(self.constant_, (1, -1))
        return self
