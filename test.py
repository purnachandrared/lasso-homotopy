import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from lasso_homotopy import OnlineLassoHomotopy

def test_collinear_features():
    """Test collinear features result in only one non-zero coefficient."""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([1, 2, 3])
    model = OnlineLassoHomotopy(lambda_=0.1)
    model.fit(X, y)
    assert sum(model.coef_ != 0) == 1, "Collinearity not handled"

def test_overfitting():
    """Test LASSO prevents overfitting in high dimensions."""
    X, y = make_regression(n_samples=50, n_features=100, noise=0.1)
    model = OnlineLassoHomotopy(lambda_=0.5)
    model.fit(X, y)
    assert sum(model.coef_ != 0) < 50, "Model overfits"

def test_large_lambda():
    """Test large lambda zeroes out all coefficients."""
    X, y = make_regression(n_samples=10, n_features=5)
    model = OnlineLassoHomotopy(lambda_=1e6)
    model.fit(X, y)
    assert np.allclose(model.coef_, 0, atol=1e-4), "Large lambda fails"

def test_online_vs_batch():
    """Test online updates match batch training."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    model_batch = OnlineLassoHomotopy(lambda_=0.1).fit(X, y)
    model_online = OnlineLassoHomotopy(lambda_=0.1).fit(X[:50], y[:50])
    for i in range(50, 100):
        model_online.update(X[i], y[i])
    assert np.allclose(model_batch.coef_, model_online.coef_, atol=1e-3)

def test_against_sklearn():
    """Validate against Scikit-Learn's Lasso."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    model = OnlineLassoHomotopy(lambda_=0.1).fit(X, y)
    sk_model = Lasso(alpha=0.1, fit_intercept=True).fit(X, y)
    assert np.allclose(model.coef_, sk_model.coef_, atol=1e-3) 
