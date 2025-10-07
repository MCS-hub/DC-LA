import numpy as np
from scipy.fftpack import dct
def l1(x):
    return np.sum(np.abs(x))

def prox_l1(x, alpha):
    # Soft-thresholding (L1 prox)
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.0)

def l2(x):
    return np.sqrt(np.sum(x**2))

def prox_l2(x, alpha):
    # L2 prox
    return np.maximum(1-alpha/np.sqrt(np.sum(x**2)), 0.0) * x


def prox_l1_minus_l2(y, alpha, epsilon=1.0):
    """
    Prox of r_epsilon(x) = ||x||_1 - epsilon * ||x||_2 at y, parameter lam > 0.
    Implements Lemma 1 (Lou & Yan, 2016).
    """
    y = np.asarray(y, dtype=float)
    amax = np.max(np.abs(y))
    if amax <= (1 - epsilon) * alpha:
        return np.zeros_like(y)

    if amax > alpha:
        z = prox_l1(y, alpha)
        nz = np.linalg.norm(z)
        # nz > 0 because amax > lam
        scale = (nz + epsilon * alpha) / nz
        return scale * z

    # (1 - alpha)*lam < amax <= lam  
    i = np.argmax(np.abs(y))
    mag = amax + (epsilon - 1) * alpha           # > 0 here
    x = np.zeros_like(y)
    x[i] = np.sign(y[i]) * mag
    return x

def compute_rela_err(x_est, x_true):  # smaller is better
    num = np.linalg.norm(x_true - x_est)**2
    den = np.linalg.norm(x_true)**2
    return num / den

def partial_dct_matrix(m, n):
    r = np.random.rand(m)  # m random frequencies in [0,1]
    A = np.zeros((m, n))
    for i in range(n):
        A[:, i] = (1 / np.sqrt(m)) * np.cos(2 * (i + 1) * np.pi * r)
    return A


# to check------
def prox_sigma_q(y, alpha, q):
    """
    Proximal operator of alpha * ||.||_{sigma_q},
    where ||x||_{sigma_q} = sum of q largest magnitudes of x.
    """
    y = np.asarray(y, dtype=float)
    abs_y = np.abs(y)
    # Indices of q largest entries
    idx = np.argsort(-abs_y)[:q]
    x = np.array(y, copy=True)
    # Apply soft-thresholding only on top-q entries
    x[idx] = np.sign(y[idx]) * np.maximum(abs_y[idx] - alpha, 0.0)
    return x


def prox_l1_minus_sigma_q(y, alpha, q):
    """
    Proximal operator of alpha * (||.||_1 - ||.||_{sigma_q}).
    Equivalent to soft-thresholding everywhere except we keep q indices
    (with largest scores) unpenalized.
    """
    y = np.asarray(y, dtype=float)
    abs_y = np.abs(y)

    # Score function: cost saved by leaving i unpenalized
    scores = np.where(
        abs_y <= alpha,
        0.5 * abs_y**2,
        alpha * abs_y - 0.5 * alpha**2
    )

    # Pick q indices with largest scores (unpenalized)
    idx = np.argsort(-scores)[:q]

    # Apply soft-thresholding everywhere
    x = prox_l1(y, alpha)
    # Restore unpenalized coordinates to original values
    x[idx] = y[idx]
    return x
