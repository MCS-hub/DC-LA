import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from utils import prox_l1_minus_l2, prox_l1, prox_l2, partial_dct_matrix
from sampling_algs import PSGLA, DC_LA, ULA
from sklearn.decomposition import PCA

np.random.seed(42)

def rel_err(xhat, xtrue):
    return np.linalg.norm(xhat - xtrue)**2 / np.linalg.norm(xtrue)**2

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


def main():
    # dimensions
    d, m, k = 2000, 200, 15
    sigma = 0.01  # noise std

    # sensing matrix (Gaussian, column norms ~1)
    # A = np.random.randn(m, d) / np.sqrt(m) 

    A = partial_dct_matrix(m, d)

    # k-sparse ground truth
    x_true = np.zeros(d)
    supp = np.random.choice(d, k, replace=False)
    x_true[supp] = np.random.randn(k)

    # data
    y = A @ x_true + sigma * np.random.randn(m)

    # Run Lasso to find an initial guess for tau
    alphas = np.logspace(-6, 0, 50)
    lcv = LassoCV(alphas=alphas, fit_intercept=False, max_iter=10000, cv=5)
    lcv.fit(A, y)


    def grad_f(x):
        return (1 / sigma**2) * A.T @ (A @ x - y)
    tau = (m / sigma**2) * lcv.alpha_  # regularize the sklearn alpha to match our formulation
    def prox_r(x, alpha):
        return prox_l1_minus_l2(x, tau * alpha)
    def prox_r1(x, alpha):
        return prox_l1(x, tau * alpha)
    def prox_r2(x, alpha):   
        return prox_l2(x, tau * alpha)
    X0 = np.zeros(d)

    # sampling
    n_samples = 50000
    burn_in = 10000
    lam = 5e-6
    gamma = 5e-7
    samples_psgla = PSGLA(X0, n_samples, burn_in, gamma, d, grad_f, prox_r)
    samples_dcla = DC_LA(X0, n_samples, burn_in, lam=lam, gamma=gamma, d=d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
    samples_ula = ULA(X0, n_samples, burn_in, lam=lam, gamma=gamma, d=d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)


    # Compare samplers on support
    samples_dict = {
        "DC-LA": samples_dcla,
        "PSGLA": samples_psgla,
        "ULA": samples_ula
    }

    plt.rcParams.update({
    "font.size": 16,        # controls default text size
    "axes.titlesize": 18,   # fontsize of axes title
    "axes.labelsize": 16,   # fontsize of x and y labels
    "xtick.labelsize": 14,  # fontsize of the x tick labels
    "ytick.labelsize": 14,  # fontsize of the y tick labels
    "legend.fontsize": 14   # fontsize of the legend
    })
    plot_ci_comparison(samples_dict, x_true, supp, 
                   "figs/l12/exp_cs/ci_support_comparison_dct.png", 
                   title_suffix="CI on Support")
    zero_indices = np.setdiff1d(np.arange(d), supp)
    plot_ci_comparison(samples_dict, x_true, zero_indices[:50], 
                    "figs/l12/exp_cs/ci_zero_comparison_dct.png", 
                    title_suffix="CI on Zero Coords (subset)")


if __name__ == "__main__":
    main()
