# ensemble_da.py
import numpy as np

def P_Gaussian(x, mu, Sigma):
    """
    Return log N(x | mu, Sigma); Sigma can be full or diagonal (np.diag(...)).
    Shapes:
      x, mu: (d,)
      Sigma: (d,d)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    S  = np.asarray(Sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Sigma must be a square covariance matrix (d x d).")
    d = x.size
    L = np.linalg.cholesky(S + 1e-12 * np.eye(d))
    z = np.linalg.solve(L, x - mu)
    maha = np.dot(z, z)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)
