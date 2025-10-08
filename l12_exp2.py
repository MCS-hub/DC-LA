import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import pickle
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from utils import l1, prox_l1, l2, prox_l2, prox_l1_minus_l2
from sampling_algs import DC_LA, ULA, PSGLA
np.random.seed(42)

# multiple chains
def main():
    d = 2
    mu_x_list = [0, 1, 2, 3]
    Sigma_x_list = [np.array([[1, 0.0], [0.0, 1]]), np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 2]])]
    tau_list = [10]

    lam, gamma = 0.01, 0.005
    n_samples = 1000
    burn_in = 0
    n_chains = 5000
    for tau in tau_list:
        for mu_x in mu_x_list:
            for Sigma_x in Sigma_x_list:
                print("mu_x:", mu_x, "Sigma_x:", Sigma_x, "tau:", tau)
                figname = f"figs/l12/exp2/dcla_mu{mu_x}_Sigma{Sigma_x[0,1]}_tau{tau}_last.png"
                def r1(x):
                    return tau * l1(x)
                def prox_r1(x, alpha):
                    return prox_l1(x, tau * alpha)
                def r2(x):
                    return tau * l2(x)
                def prox_r2(x, alpha):
                    return prox_l2(x, tau * alpha)
                def prox_r(x, alpha):
                    return prox_l1_minus_l2(x, tau * alpha)
                def f(x):
                    return 0.5 * (x-mu_x).T @ Sigma_x @ (x-mu_x)  
                def grad_f(x):
                    return Sigma_x @ (x -mu_x) 
                def V(x):
                    return f(x) + r1(x) - r2(x)
                
                samples_ula_last = []
                samples_dcla_last = []
                samples_psgla_last = []

                def run_chain_once(d, n_samples, burn_in, lam, gamma, grad_f, prox_r1, prox_r2, prox_r):
                    X0 = np.random.randn(d)
                    samples_ula  = ULA(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                    samples_dcla = DC_LA(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                    samples_psgla = PSGLA(X0, n_samples, burn_in, gamma, d, grad_f=grad_f, prox_r=prox_r)
                    return samples_ula[-1], samples_dcla[-1], samples_psgla[-1]

                # Run chains in parallel
                results = Parallel(n_jobs=-1)(   # -1 = use all available cores
                    delayed(run_chain_once)(d, n_samples, burn_in, lam, gamma, grad_f, prox_r1, prox_r2, prox_r)
                    for _ in range(n_chains)
                )

                # Unpack results
                samples_ula_last, samples_dcla_last, samples_psgla_last = zip(*results)

                samples_ula_last = np.array(samples_ula_last)
                samples_dcla_last = np.array(samples_dcla_last)
                samples_psgla_last = np.array(samples_psgla_last)
                # target density
                def pi_unnormalized(x):
                    return np.exp(-V(x))

                def pi_unnormalized_2d(x1, x2):
                    x = np.array([x1, x2])
                    return pi_unnormalized(x)

                if mu_x > 1.5:
                    xlim1 = 1
                    xlim2 = 7
                else:
                    xlim1 = 3
                    xlim2 = 3

                x1 = np.linspace(-xlim1, xlim2, 300)
                x2 = np.linspace(-xlim1, xlim2, 300)
                # Integral 
                val, err = dblquad(lambda x2, x1: pi_unnormalized_2d(x1, x2),
                                -xlim1, xlim2,  # x1 range
                                lambda _: -xlim1, lambda _: xlim2)  # x2 range

                def pi_normalized(x):
                    return np.exp(-V(x))/val

                # plot
                # Meshgrid
                X1, X2 = np.meshgrid(x1, x2)
                f_vec = np.vectorize(lambda x, y: pi_normalized(np.array([x, y])))
                Z = f_vec(X1, X2)

                # Bins
                xbin = np.linspace(-xlim1, xlim2, 100)
                ybin = np.linspace(-xlim1, xlim2, 100)

                vmin = float(np.min(Z))
                vmax = float(np.max(Z))

                H_dcla, _, _ = np.histogram2d(samples_dcla_last[:,0], samples_dcla_last[:,1],
                                                bins=[xbin, ybin], density=True)
                H_ula,  _,  _  = np.histogram2d(samples_ula_last[:,0],  samples_ula_last[:,1],
                                                bins=[xbin, ybin], density=True)
                H_psgla,_,  _  = np.histogram2d(samples_psgla_last[:,0], samples_psgla_last[:,1],
                                                bins=[xbin, ybin], density=True)
                vmin = min(vmin, H_dcla.min(), H_ula.min(), H_psgla.min())
                vmax = max(vmax, H_dcla.max(), H_ula.max(), H_psgla.max())
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = 'viridis'

                fig, axs = plt.subplots(2, 2, figsize=(8, 8))

                # --- Subplot 1: Contour ---
                #levels = np.linspace(vmin, vmax, 8)
                cs = axs[0,0].contourf(X1, X2, Z, norm=norm, cmap=cmap)  # levels enforce same scale
                axs[0,0].set_title("Target Distribution")
                #fig.colorbar(cs, ax=axs[0,0])

                # --- Subplot 2: DC-LA ---
                H, xedges, yedges, im1 = axs[0,1].hist2d(samples_dcla_last[:,0], samples_dcla_last[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[0,1].set_title("DC-LA")
                #fig.colorbar(im1, ax=axs[0,1])

                # --- Subplot 3: ULA ---
                H, xedges, yedges, im2 = axs[1,1].hist2d(samples_ula_last[:,0], samples_ula_last[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[1,1].set_title("ULA")
                #fig.colorbar(im2, ax=axs[1,1])

                # --- Subplot 4: PSGLA ---
                H, xedges, yedges, im3 = axs[1,0].hist2d(samples_psgla_last[:,0], samples_psgla_last[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[1,0].set_title("PSGLA")

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im3, cax=cbar_ax)

                # Layout adjustments
                #plt.tight_layout()
                plt.savefig(figname, bbox_inches="tight", dpi=600)

                # results = {
                #     "samples_dcla_last": samples_dcla_last,
                #     "samples_ula_last": samples_ula_last,
                #     "samples_psgla_last": samples_psgla_last,
                #     "Z": Z,
                #     "mu_x": mu_x,
                #     "Sigma_x": Sigma_x,
                #     "tau": tau,
                # }

                # # Save to pickle
                # fname = f"results/l12/exp2/exp2_results_mu{mu_x}_Sigma{Sigma_x[0,1]}_tau{tau}.pkl"
                # with open(fname, "wb") as f:
                #     pickle.dump(results, f)
                
if __name__ == "__main__":
    main()
