import numpy as np

class OnlineLassoHomotopy:
    """
    LASSO Regression with Homotopy Method for Online Updates.
    
    Attributes:
        lambda_ (float): Regularization strength.
        coef_ (np.ndarray): Model coefficients.
        intercept_ (float): Model intercept.
        XtX (np.ndarray): Sufficient statistic (covariance matrix).
        Xty (np.ndarray): Sufficient statistic (cross-correlation vector).
        active_set (list): Indices of active (non-zero) coefficients.
    """
    
    def __init__(self, lambda_=1.0, tol=1e-4):
        self.lambda_ = lambda_
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.XtX = None
        self.Xty = None
        self.n_samples = 0
        self.active_set = []
        self.signs = []
        self.X_mean_ = None
        self.X_std_ = None
        self.y_mean_ = None

    def _preprocess(self, X, y=None):
        """Center and scale features/target."""
        if self.n_samples == 0:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_ = np.std(X, axis=0)
            self.X_std_[self.X_std_ == 0] = 1.0
            if y is not None:
                self.y_mean_ = np.mean(y)
        X = (X - self.X_mean_) / self.X_std_
        return X, (y - self.y_mean_ if y is not None else None)

    def fit(self, X, y):
        """Fit model to initial batch of data."""
        X, y = self._preprocess(X, y)
        self.n_samples = X.shape[0]
        self.XtX = X.T @ X
        self.Xty = X.T @ y
        self.coef_ = np.zeros(X.shape[1])
        self._homotopy_loop()
        self.intercept_ = self.y_mean_ - (self.X_mean_ / self.X_std_) @ self.coef_
        return self

    def update(self, x_new, y_new):
        """Update model with a new observation."""
        x_new, y_new = self._preprocess(x_new.reshape(1, -1), y_new)
        self.XtX += np.outer(x_new, x_new)
        self.Xty += x_new.flatten() * y_new
        self.n_samples += 1
        self._homotopy_loop()
        self.intercept_ = self.y_mean_ - (self.X_mean_ / self.X_std_) @ self.coef_
        return self

    def _homotopy_loop(self):
        """Core homotopy algorithm to update coefficients."""
        # Implementation details from earlier steps
        pass  # (Full code provided in previous responses)

    def predict(self, X):
        """Make predictions."""
        X = (X - self.X_mean_) / self.X_std_
        return X @ self.coef_ + self.intercept_
