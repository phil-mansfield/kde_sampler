import kde_sampler
import numpy.random as random
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

import palette
from palette import pc

def main():
    gen = random.default_rng()

    dim = 3
    
    kernel = kde_sampler.M4Kernel(dim=dim)
    n_p = int(10**4)
    n_s = int(10**6)
    
    sim_str = "p%d_s%d_d%d_mixed" % (np.log10(n_p), np.log10(n_s), dim)

    x_p = gen.random(size=(n_p,dim))
    m_p = np.ones(n_p)

    #x_p = np.vstack([gen.random(size=(n_p,dim)), gen.random(size=(10*n_p,dim))])
    #m_p = np.hstack([np.ones(n_p), np.ones(10*n_p)/100])

    plt.figure()

    ks = [2, 4, 8, 16, 32, 64, 128]
    colors = [pc("a"), pc("r"), pc("o"), pc("g"), pc("b"), pc("p"), pc("k")]

    ratios = [2, 4, 8, 16]
    n_cutoffs = np.zeros((len(ratios), len(ks)))

    for j in range(len(ks)):
        sampler = kde_sampler.KDESampler(
            kernel, ks[j], x_p, m_p, boxsize=1)
        x = sampler.sample(n_s)
    
        #bins = np.array(10**np.linspace(1, 2.5, 30), dtype=int)
        bins = np.array(10**np.linspace(1, 2.5, 30), dtype=int)
        var_mean = np.zeros(len(bins))

        for i in range(len(bins)):
            n = density_pdf(x, bins[i])
            var_mean[i] = np.var(n)/np.mean(n)

        f = interpolate.interp1d(np.log10(var_mean), np.log10(n_s/bins**dim))
        n_cutoffs[:,j] = 10**f(np.log10(ratios))

        plt.plot(len(x)/bins**3, var_mean,
            c=colors[j], label=r"$k=%d$" % ks[j])

    plt.legend(loc="upper left", fontsize=16, frameon=True)
    plt.xscale("log")
    plt.yscale("log")

    lo, hi = plt.xlim()
    plt.xlim(lo, hi)
    plt.plot([lo, hi], [1, 1], "--", c=pc("a"), lw=1.5)
    plt.plot([lo, hi], [n_s/n_p, n_s/n_p], "--", c=pc("a"), lw=1.5)
    plt.xlabel(r"$\langle n_\star\rangle$")
    plt.ylabel(r"$\sigma^2(n_\star)/\langle n_\star\rangle$")
    plt.ylim(0.8, 3*n_s/n_p)

    plt.savefig("plots/var_mean_%s.png" % (sim_str))

    plt.figure()

    for i in range(len(ratios)):
        plt.plot(n_cutoffs[i,:], ks, color=colors[1:][i],
            label=r"${\rm var(n)/n < %d}$" %ratios[i])
    plt.xlabel(r"$\langle n_\star\rangle$")
    plt.ylabel(r"$k$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", fontsize=16)

    plt.savefig("plots/n_k_%s.png" % (sim_str))

    plt.figure()

    for i in range(len(ratios)):
        ns_inside_k = np.array(ks)*(n_s/n_p)
        plt.plot(n_cutoffs[i,:], (ns_inside_k/n_cutoffs[i,:])**(1/3),
            color=colors[1:][i],
            label=r"${\rm var(n)/n < %d}$" %ratios[i])
    plt.xlabel(r"$\langle n_\star\rangle$")
    plt.ylabel(r"$R_k/R_{\rm cutoff}$")
    plt.xscale("log")
    plt.legend(loc="upper right", fontsize=16)

    plt.savefig("plots/rk_rc_%s.png" % (sim_str))

    plt.show()

def density_pdf(x, bins):
    rng = [(0, 1) for _ in range(x.shape[1])]
    n, _ = np.histogramdd(x, bins, range=rng)
    return n.flatten()

if __name__ == "__main__": main()
