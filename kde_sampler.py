import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import numpy.random as random
import scipy.integrate as integrate
import numpy.random as random
import matplotlib.colors as colors
import time
import abc

# This just loads my preferred matplotlib config file and color scheme
import palette
palette.configure(False)
from palette import pc
# You can comment those three lines out and define `pc = lambda x: x` if you
# don't have the library.

def main():
    #interpolate_test()
    #m4_kernel_test()
    #m4_sample_test()
    kde_sampler_test()
    plt.show()

    #benchmark_kde_sampler()


class Kernel(abc.ABC):
    """ Kernel is an abstract class representing a KDE kernel. To make a
    new kernel, inheret this class and implement w_exact(q).
    """
    
    def __init__(self, dim=3, table_size=1<<12):
        """ dim is the kernel's dimensionality, table_size is the
        size od internal interpolation tables.
        """
        # A table size of 1<<10 leads to interpolation errors smaller than
        # one part in 10^6 on average, so 1<<12 is safe.
        self.n = table_size
        self.dx = 1/self.n
        self.dim = dim

        self._q = np.linspace(0, 1, table_size+1)
        self._w = np.array([self.w_exact(q) for q in self._q])
        self._m = np.array([self.m_exact(q) for q in self._q])

        self._m_inv = np.linspace(0, 1, table_size+1)
        self._q_inv = interpolate(self._m, self._q, -1, self._m_inv)

    @abc.abstractmethod
    def w_exact(self, q):
        """ w_exact is an abstract method that computes the window function
        at q = r/h, where h is the kernel radius. It is okay for this function
        to be slow and unvectorized.
        """
        pass

    def m_exact(self, q):
        """ m_exact computes the fraction of a kernel's mass contained within
        q = r/h, where h is the kernel radius. This function is slow and
        unvectorized.
        """
        num = integrate.quad(lambda x: x**(self.dim-1) *
            self.w_exact(x), 0, q)[0]
        den = integrate.quad(lambda x: x**(self.dim-1) *
            self.w_exact(x), 0, 1)[0]
        return num/den

    def w(self, q):
        """ w computes the window function at q = r/h, where h is the kernel
        radius. This function is interpolated and fast.
        """
        return interpolate(self._q, self._w, self.dx, q)

    def m(self, q):
        """ m computes the fraction of the kernel's mass that's contained
        within q = r/h, where h is the kernel radius.
        """
        return interpolate(self._q, self._m, self.dx, q)

    def m_inv(self, m):
        """ m_inv computes the the inverse of the m method.
        """
        return interpolate(self._m_inv, self._q_inv, self.dx, m)

    def sample(self, gen, n):
        """ sample uses gen, a numpy.random.Generator, to generate n samples
        from the kernel. Works in any number of dimensions and uses specialized
        routines in lower dimensions.
        """
        r = self.m_inv(gen.random(n))
        if self.dim == 1:
            sgn = 2*gen.integers(2, size=n) - 1
            return r*sgn
        elif self.dim == 2:
            # We can save an RNG call over the general case.
            phi = 2*np.pi*gen.random(size=n)

            out = np.zeros((len(r), 2))
            out[:,0] = r*np.sin(phi)
            out[:,1] = r*np.cos(phi)
        elif self.dim == 3:
            # We can save an RNG call over the general case.
            # (...but maybe it's not worth it?)
            phi = 2*np.pi*gen.random(size=n)
            theta = np.arccos(2*gen.random(size=n) - 1)

            out = np.zeros((len(r), 3))
            out[:,0] = r*np.sin(theta)*np.cos(phi)
            out[:,1] = r*np.sin(theta)*np.sin(phi)
            out[:,2] = r*np.cos(theta)
        else:
            # Hypersphere. See summary of Marsaglia (1972) here
            # https://mathworld.wolfram.com/HyperspherePointPicking.html
            # (But it's just taking advantage of the fact that Guassians
            # are rotationally symmetrical.)

            x = gen.normal(size=(n, self.dim))
            r0 = np.sqrt(np.sum(x*x, axis=1))

            out = np.zeros((len(r), self.dim))
            for i in range(out.shape[1]):
                out[:,i] = r/r0*x[:,i]

        return out

class M4Kernel(Kernel):
    """ M4Kernel implements an M_4 KDE kernel. See section 4.2 of Springel
    et al. (2021) for some discussion.
    """
    def w_exact(self, q):
        """ w_exact computes the window function at q = r/h, where h is the
        kernel radius.
        """
        if q < 0.5:
            return 1 - 6*q**2 + 6*q**3
        else:
            return 2*(1 - q)**3

def interpolate(x, y, dx, xi):
    """ interpolate linearly interpolates a table defined by the arrays x and
    y at xi. x must be sorted from smallest to largest. If x is uniformly
    spaced, you can substantially accelerate interpolation by passing the
    spacing at dx. Otherwise, set dx to -1.
    """
    if dx < 0:
        i_int = np.searchsorted(x, xi)
        i_int[i_int == 0] += 1
        i_int -= 1
        dx = x[i_int+1] - x[i_int]
        i_frac = (xi - x[i_int])/dx
    else:
        i_full = xi/dx
        i_int = np.array(xi/dx, dtype=int)
        i_int[i_int == len(x) - 1] = len(x) - 2
        i_frac =  i_full - i_int


    return y[i_int] + i_frac*(y[i_int+1] - y[i_int])

class KDESampler(object):
    """ KDESampler smooths a density field represented by a set of points
    of variable mass. This smooth density field can be sampled efficiently.
    """
    def __init__(self, kernel, k, x, m,
                 method="split", gen=random.default_rng(),
                 boxsize=None):
        """
        kernel - the KDE kernel, must be an instance of
        k - number of neighbors to smooth over
        x - particle positions; has shape (n, dim)
        m - particle masses
        method - method to use when computing KDE radii. "standard" sets
        each KDE radius to the kth-nearest neighbor and "split" sets each KDE
        radius to the kth-nearest neighbot with exactly the same mass as the
        particle.
        gen - the RNG, must be a numpy.random.Generator
        boxsize - the size of the periodic box, if non-periodic, set to None
        """
        # Could be extended so that secondary star particle properties
        # are passed to __init__ and are sampled via per-particle
        # covariance matrices.

        self.kernel, self.gen, self.method = kernel, gen, method
        self.k = k

        # Sort by mass and compute edge locations.
        order = np.argsort(m)
        x, m = x[order], m[order]
        edges = np.where(m[1:] != m[:-1])[0]
        edges = np.hstack([[0], edges + 1, [len(x)]])

        # Index accounting
        self.starts, self.ends = edges[:-1], edges[1:]
        self.counts = self.ends - self.starts
        self.x, self.m = x, m
        self.mass_bins = self.m[self.starts]

        # Probability of each mass bin being chosen
        self.bin_weights = self.mass_bins*self.counts
        self.bin_weights /= np.sum(self.bin_weights)

        # This could be exteneded to compute covariance matrices at each
        # point.
        if method == "standard":
            tree = spatial.KDTree(x, boxsize=boxsize)
            self.r = tree.query(x, k+1)[0][:,k]
        elif method == "split":
            self.r = np.zeros(len(x))

            for i in range(len(self.mass_bins)):
                xi = self.x[self.starts[i]: self.ends[i]]

                tree = spatial.KDTree(xi, boxsize=boxsize)
                self.r[self.starts[i]: self.ends[i]] = tree.query(
                    xi, k+1)[0][:,k]
        elif method == "mass":
            assert(0)
        else:
            assert(0)

    def sample(self, n_avg):
        """ sample samples KDESampler's underlying PDF. n_avg is the
        expected number of samples. The actual number of samples will follow
        a Poisson distribution around this average.
        """
        # Can be modified to allow for exact sampling if wanted. If you
        # generate too many points, you can clip the output array to have
        # length n_avg after shuffling. If you generate too many, you need
        # to generate more and hstack the arrays together. It's probably
        # best ot internally overestimate the number of particles you need by
        # a couple sigma in that case.

        n_bins = len(self.mass_bins)
        counts = self.gen.poisson(n_avg * self.bin_weights)
        ends = np.cumsum(counts)
        starts = ends - counts

        out = np.zeros((np.sum(counts), self.kernel.dim))
        for i in range(n_bins):
            # Start and end of each mass bin in the input array
            start_i, end_i = self.starts[i], self.ends[i]
            # Start and end of each mass bin in the output array
            start_o, end_o = starts[i], ends[i]

            idx = self.gen.integers(start_i, end_i, counts[i])
            xr = self.kernel.sample(self.gen, counts[i])

            for dim in range(self.kernel.dim):
                offset = xr[:,dim]*self.r[idx]
                out[start_o:end_o,dim] = self.x[idx,dim] + offset

        self.gen.shuffle(out)
        return out
        



#######################
## testing functions ##
#######################

def interpolate_test():
    """ Creates a plot which compares two damped sign waves to coarse
    interpolations made with differetn spacings of the interpolation points.
    Check that they look right by eye.
    """
    random.seed(0)
    plt.figure()

    def f(x): return np.sin(x*4)*np.exp(-x)

    x = np.linspace(0, 4, 300)
    x1 = np.linspace(0, 4, 20)
    dx = x1[1] - x1[0]
    y1 = f(x1)

    plt.plot(x, f(x), c=pc("r"))
    plt.plot(x1, y1, "o", c="k")

    xi = 4*random.random(1000)
    xi[0] = 0
    x1[1] = 4
    yi = interpolate(x1, y1, dx, xi)

    plt.plot(xi, yi, ".", c="k", alpha=0.2)

    x1 = np.zeros(20) 
    x1[-1] = 4
    x1[1:-1] = np.sort(random.random(18))*4
    y1 = f(x1)

    plt.plot(x, f(x) + 1, c=pc("b"))
    plt.plot(x1, y1 + 1, "o", c="k")

    yi = interpolate(x1, y1, -1, xi)

    plt.plot(xi, yi + 1, ".", c="k", alpha=0.2)

    plt.text(2, 1.5, "Non-uniform")
    plt.text(2, 0.5, "Uniform")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

def m4_kernel_test():
    """ Overplots exact values of kernels in different dimensions with 
    interpolated tables (and inverse interpolation for mass). Check that
    all the curves line up.
    """
    k_1d = M4Kernel(1, 1000)
    k_2d = M4Kernel(2, 1000)
    k_3d = M4Kernel(3, 1000)

    q = np.linspace(0, 1, 200)

    w_exact, w = np.zeros(len(q)), np.zeros(len(q))
    for i in range(len(q)):
        w_exact[i], w[i] = k_1d.w_exact(q[i]), k_1d.w(q[i])

    plt.figure()

    plt.plot(q, w_exact, c="k", label=r"${\rm exact}$")
    plt.plot(q, w, "--", c=pc("r"), label=r"${\rm interpolated}$")
    plt.xlabel("$r/h$")
    plt.ylabel("$w_{M4}(r)$")
    plt.legend(loc="upper right")

    
    plt.figure()
    for off, k in enumerate([k_1d, k_2d, k_3d]):
        m, m_exact = np.zeros(len(q)), np.zeros(len(q))
        m_inv = np.zeros(len(q))
        for i in range(len(q)):
            m_exact[i], m[i] = k.m_exact(q[i]), k.m(q[i])
            m_inv[i] = k.m_inv(q[i])
        if off == 0:
            plt.plot(q, m_exact+off, c="k",
                label=r"${\rm exact}$")
            plt.plot(q, m_exact+off, "--", c=pc("r"),
                label=r"${\rm interpolated}$")
            plt.plot(m_inv, q+off, ":", c=pc("b"),
                label=r"${\rm inverted}$")
        else:
            plt.plot(q, m_exact+off, c="k")
            plt.plot(q, m_exact+off, "--", c=pc("r"))
            plt.plot(m_inv, q+off, ":", c=pc("b"))
        plt.text(0.0, off+0.3, r"${\rm %dD}$" % (off+1))

    plt.xlabel(r"$r/h$")
    plt.ylabel(r"$m(<r)$")
    plt.legend(loc="lower right")

    plt.figure()

    n = 2**np.linspace(0, 12, 12, dtype=int)
    var = np.zeros(len(n))
    xi = random.random(1000)
    for i in range(len(n)):
        k = M4Kernel(3, n[i])
        m_exact = np.zeros(len(xi))
        for j in range(len(xi)):
            m_exact[j] = k.m_exact(xi[j])
        m = k.m(xi)

        var[i] = np.mean((m-m_exact)**2)

    plt.plot(n, np.sqrt(var), c=pc("r"))
    plt.xscale("log")
    plt.yscale("log")
    lo, hi = plt.ylim()
    plt.ylim(lo, hi)
    plt.plot([1e3, 1e3], [lo, hi], "--", c="k", lw=2,
             label=r"${\rm default}$")

    plt.legend(loc="lower left")
    plt.xlabel(r"${\rm table\ size}$")
    plt.ylabel(r"$\sqrt{{\rm var}(m)}$")
 
def m4_sample_test():
    """ Generates tons of samples from the m4 kernel in different
    dimensions. Makes a histogram in 1d, an impage in 2d and a pair
    of images along different directions in 3d. This one isn't fancy,
    it only catches errors that totally brick the sampler (i.e. most possible
    issues) so just see if these things look nice.
    """
    n = int(1e7)

    gen = random.default_rng(seed=0)

    x_1d = M4Kernel(dim=1).sample(gen, n)
    x_2d = M4Kernel(dim=2).sample(gen, n)
    x_3d = M4Kernel(dim=3).sample(gen, n)

    plt.figure()
    plt.title("1D")
    plt.hist(x_1d, histtype="step", bins=300, lw=3, color=pc("r"))

    plt.figure()
    plt.title("2D")
    plt.hexbin(x_2d[:,0], x_2d[:,1], gridsize=60)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("3D")
    ax[0].hexbin(x_3d[:,0], x_3d[:,1], gridsize=60)
    ax[1].hexbin(x_3d[:,0], x_3d[:,2], gridsize=60)

    plt.figure()
    plt.hist(np.sqrt(np.sum(x_3d*x_3d, axis=1)),
        histtype="step", bins=301, lw=3, color=pc("r"))

def kde_sampler_test():
    """ Generates three plots. The first shows a set of points and the standard
    KDE smoothing of those points with k=64 in 2D. The second shows what those
    KDE PDFs look like if the particles are embedded in a bath of many low-mass
    particles. The left panel of this second plot shows standard KDE, the right
    shows KDEs that only look for equal-mass particles. The third plot shows the
    same, except it isn't log-scaled.
    """
    gen = random.default_rng(seed=0)

    kernel = M4Kernel(dim=2)

    n1 = int(1e3)
    x1 = gen.normal(size=(n1, 2))
    m1 = np.ones(n1)

    sampler = KDESampler(kernel, 64, x1, m1, method="standard")

    xs1 = sampler.sample(int(1e6))

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].plot(x1[:,0], x1[:,1], "o", c=pc("r"))
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)

    ax[1].hexbin(xs1[:,0], xs1[:,1], extent=[-3, 3, -3, 3],
        norm=colors.LogNorm())

    ####

    n2 = int(1e4)
    x2 = gen.normal(size=(n2, 2)) * 2
    m2 = np.ones(n2) * 0.01

    x = np.vstack([x1, x2])
    m = np.hstack([m1, m2])

    _, ax1 = plt.subplots(1, 2, figsize=(14, 7))
    _, ax2 = plt.subplots(1, 2, figsize=(14, 7))

    sampler = KDESampler(kernel, 64, x, m, method="standard")
    xs = sampler.sample(int(1e6))
    ax1[0].hexbin(xs[:,0], xs[:,1], extent=[-3, 3, -3, 3],
        norm=colors.LogNorm())
    ax2[0].hexbin(xs[:,0], xs[:,1], extent=[-3, 3, -3, 3])

    sampler = KDESampler(kernel, 64, x, m, method="split")
    xs = sampler.sample(int(1e6))
    ax1[1].hexbin(xs[:,0], xs[:,1], extent=[-3, 3, -3, 3],
        norm=colors.LogNorm())
    ax2[1].hexbin(xs[:,0], xs[:,1], extent=[-3, 3, -3, 3])

def benchmark_kde_sampler():
    """ Prints out benchmarks for generating the underling KD tree and for
    sampling it.
    """
    gen = random.default_rng()

    dim = 6
    k = 64
    n = [int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6]]
    kernel = M4Kernel()

    n_sample = 1e7
    
    print("In %d dimensions with k = %d" % (dim, k))
    
    for i in range(len(n)):

        x = gen.normal(size=(n[i],dim))
        m = np.ones(len(x))
        
        t1 = time.time()
        sampler = KDESampler(kernel, k, x, m)
        t2 = time.time()
        _ = sampler.sample(n_sample)
        t3 = time.time()
        
        print("n = %.1g:" % n[i])
        print("    dt tree:   %.2f s" % (t2 - t1))
        print("    dt sample: %.2f s (n = %.1g)" % (t3 - t2, n_sample))
    
    x = gen.normal(size=())
    
if __name__ == "__main__": main()
