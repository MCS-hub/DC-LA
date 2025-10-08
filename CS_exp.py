import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import argparse
from utils import oversampled_dct_matrix, prox_l1_minus_l2, prox_l1, prox_l2, partial_dct_matrix, oversampled_dct_matrix, generate_sparse_vector, plot_ci_comparison 
from sampling_algs import PSGLA, DC_LA, ULA
from sklearn.decomposition import PCA

np.random.seed(42)

parser =argparse.ArgumentParser()
parser.add_argument('--sensing', type=str, default='pdct', choices=['pdct', 'odct', 'gaussian'], help='type of sensing matrix')
parser.add_argument('--oversample_factor', type=int, default=5, help='oversampling factor for oversampled DCT')
args = parser.parse_args()
sensing = args.sensing
F = args.oversample_factor

def main():
    # dimensions
    d, m, k = 2000, 200, 15
    sigma = 0.01  # noise std


    if sensing == 'gaussian':
        A = np.random.randn(m, d) / np.sqrt(m)
        x_true = np.zeros(d)
        supp = np.random.choice(d, k, replace=False)
        x_true[supp] = np.random.randn(k)
        fig_name_support = "figs/l12/exp_cs/ci_support_comparison_gaussian.png"
        fig_name_zero = "figs/l12/exp_cs/ci_zero_comparison_gaussian.png"
    elif sensing == 'pdct':
        A = partial_dct_matrix(m, d)
        # k-sparse ground truth
        x_true = np.zeros(d)
        supp = np.random.choice(d, k, replace=False)
        x_true[supp] = np.random.randn(k)
        fig_name_support = "figs/l12/exp_cs/ci_support_comparison_pdct.png"
        fig_name_zero = "figs/l12/exp_cs/ci_zero_comparison_pdct.png"
    elif sensing == 'odct':
        L = 2 * F
        A = oversampled_dct_matrix(m, d, F)
        x_true, supp = generate_sparse_vector(d, k, L)
        fig_name_support = f"figs/l12/exp_cs/ci_support_comparison_overdctF{F}.png"
        fig_name_zero = f"figs/l12/exp_cs/ci_zero_comparison_overdctF{F}.png"
    else:
        raise ValueError("Unknown sensing type")

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
    "font.size": 16,       
    "axes.titlesize": 18,   
    "axes.labelsize": 16,  
    "xtick.labelsize": 14, 
    "ytick.labelsize": 14,  
    "legend.fontsize": 14
    })

    
    plot_ci_comparison(samples_dict, x_true, supp, 
                   fig_name_support, 
                   title_suffix="CI on Support")
    zero_indices = np.setdiff1d(np.arange(d), supp)
    
    plot_ci_comparison(samples_dict, x_true, zero_indices[:50], 
                    fig_name_zero, 
                    title_suffix="CI on Zero Coords (subset)")

if __name__ == "__main__":
    main()
