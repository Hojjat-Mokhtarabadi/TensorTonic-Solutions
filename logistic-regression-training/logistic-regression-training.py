import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    
    n_samples = X.shape[0]
    n_feats = X.shape[1]
    
    # init weights and biases with zero
    w = np.zeros((n_feats, 1))
    b = np.zeros((1,))
    
    for step in range(steps):
        z = X @ w + b
        y_hat = _sigmoid(z)
        
        dw = X.T @ (y_hat - y) / n_samples
        db = np.mean(y_hat - y)

        w = w - lr * dw
        b = b - lr * db

    return (w.ravel(), b.item())

        
        
    