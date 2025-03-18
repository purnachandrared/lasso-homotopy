# LASSO Regression with Homotopy Method

[![PyPI](https://img.shields.io/pypi/v/numpy)](https://pypi.org/project/numpy/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An efficient implementation of LASSO regression using the Homotopy Method, supporting online updates.

## Features
- **Batch Training**: Fit model to static datasets.
- **Online Updates**: Incremental updates with new observations.
- **Sparse Solutions**: Automatically zeroes irrelevant features.

## Installation
```bash
git clone https://github.com/purnachandrared/lasso-homotopy.git
cd lasso-homotopy
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Usage
### Batch Mode
```python
from lasso_homotopy import OnlineLassoHomotopy

X, y = load_your_data()  # Replace with your data
model = OnlineLassoHomotopy(lambda_=0.1)
model.fit(X, y)
print("Coefficients:", model.coef_)
```

### Online Mode
```python
model = OnlineLassoHomotopy(lambda_=0.1)
model.fit(X_initial, y_initial)

# Update with new data points
for x_new, y_new in stream_of_data:
    model.update(x_new, y_new)
```

## Testing
```bash
pytest test_lasso.py -v
```

## Comparison with Scikit-Learn
We validate correctness against `sklearn.linear_model.Lasso`:
```python
from sklearn.linear_model import Lasso

# Results should match closely
sk_model = Lasso(alpha=0.1).fit(X, y)
your_model = OnlineLassoHomotopy(lambda_=0.1).fit(X, y)
```

## License
MIT
