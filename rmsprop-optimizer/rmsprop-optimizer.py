import numpy as np

def rmsprop_step(
    w: np.ndarray, 
    g: np.ndarray, 
    s: np.ndarray, 
    lr: float = 0.001, 
    beta: float = 0.9, 
    eps: float = 1e-8
):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w)
    s = np.asarray(s)
    g = np.asarray(g)

    assert w.ndim == s.ndim and w.ndim == g.ndim, "The dimension of w, g and s should be the same!"

    s_new = beta * s + (1-beta) * np.square(g)
    lr_coeff = lr / (np.sqrt(s_new + eps))
    w_new = w - lr_coeff * g

    return w_new, s_new
    