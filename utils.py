import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
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
    Prox of r_alpha(x) = ||x||_1 - alpha * ||x||_2 at y, parameter lam > 0.
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

def oversampled_dct_matrix(m, n, F):
    r = np.random.rand(m)  # m random frequencies in [0,1]
    A = np.zeros((m, n))
    for i in range(n):
        A[:, i] = (1 / np.sqrt(m)) * np.cos(2 * (i + 1) * np.pi * r / F)
    return A


def generate_sparse_vector(n, k, L):
    """
    Generate a k-sparse vector x in R^n with minimum separation L between spikes.

    Parameters
    ----------
    n : int
        Length of the vector.
    k : int
        Number of nonzero entries (sparsity level).
    L : int
        Minimum separation between nonzero indices.

    Returns
    -------
    x : np.ndarray
        The k-sparse vector of length n.
    support : np.ndarray
        Indices of nonzero elements.
    """
        
    if (k - 1) * L + 1 > n:
        raise ValueError("Infeasible: vector too short for given k and L.")
        
    # Available positions
    available = list(range(n))
    support = []
    
    while len(support) < k:
        idx = np.random.choice(available)
        support.append(idx)
        # Remove indices within L distance
        available = [i for i in available if abs(i - idx) >= L]
    
    x = np.zeros(n)
    x[support] = np.random.randn(k)
    support.sort()
    return x, np.array(support)


def plot_ci_comparison(samples_dict, x_true, indices, save_path, title_suffix=""):
    """
    Plot 95% confidence intervals across given indices
    for multiple samplers side by side.
    
    samples_dict: dict of {label: samples}
    x_true: true vector
    indices: list/array of dimensions to plot
    """
    dims = np.arange(len(indices))

    fig, axes = plt.subplots(1, len(samples_dict), figsize=(18, 5), sharey=True)

    for ax, (label, samples) in zip(axes, samples_dict.items()):
        lower = np.percentile(samples[:, indices], 2.5, axis=0)
        upper = np.percentile(samples[:, indices], 97.5, axis=0)
        mean_est = np.mean(samples[:, indices], axis=0)
        true_vals = x_true[indices]

        ax.fill_between(dims, lower, upper, color="skyblue", alpha=0.4, label="95% CI")
        ax.plot(dims, mean_est, "b--", label="Posterior mean")
        ax.plot(dims, true_vals, "r-", label="True x*")

        ax.set_title(f"{label} {title_suffix}")
        ax.set_xlabel("Index within selected set")

    axes[0].set_ylabel("Value")
    axes[0].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
