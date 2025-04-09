import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from scipy import special
from scipy.optimize import minimize

def getExtremeEigenvalues(M):
    """
    Compute the largest and smallest eigenvalues of a symmetric matrix using ARPACK.

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.

    Returns
    -------
    (a, b) : tuple of float
        Pair containing the smallest (a) and largest (b) eigenvalue.
    """

    smallest_eigen = eigsh(M, k=1, which='SA')[0]
    largest_eigen = eigsh(M, k=1, which='LA')[0]
    return (smallest_eigen[0],largest_eigen[0])

def normalizeEigen(M,smallest_ev,largest_ev):
    """
    Normalize the eigenvalues of a symmetric matrix between -1 and 1.

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.
    smallest_ev : float
        Smallest eigenvalue of M.
    largest_ev : float
        Largest eigenvalue of M.

    Returns
    -------
    M_norm : ndarray
        Normalized matrix.
    """
    n,_ = M.shape
    M_norm = (2*M - (smallest_ev+largest_ev)*np.eye(n))/(largest_ev - smallest_ev)
    return M_norm


def stochasticTraceEstimator(M,l,n_v = 100):
    """
    Stochastic trace estimator of moments from 0 to l.

    Parameters
    ----------
    M : ndarray
        Square symmetric matrix.
    l : int
        Largest moment to compute.
    n_v : int
        Number of random vectors to use.

    Returns
    -------
    moments : ndarray
        The first l moments of M.
    """
    n,_ = M.shape
    # generate ramdom vectors
    G = np.random.randn(n,n_v)
    # normalize each column
    norms = np.linalg.norm(M,axis=0,keepdims=True)
    # normalize columns
    G_norm = G/norms
    # create list where the moments will be created
    moments = []
    curr_MG = G_norm.copy()
    for _ in range(l+1):
        curr_moment = np.mean(np.sum(G * curr_MG,axis = 0))
        moments.append(curr_moment)
        # update (M^k)G
        curr_MG = M @ curr_MG
    # 
    return moments


def ChebyshevSpectralDensity(M,p = 50,n_v = 100,npoints=512):
    """
    Chebyshev method to obtain the spectral density using Kernel Polynomial Method (KPM).

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.
    p : int
        Chebyshev polynomial degree.
    n_v : int
        Number of random vectors.
    npoints : int
        Number of discretization points.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    n,_ = M.shape
    # obtain the smallest and largest eigenvalue
    smallest_ev,largest_ev = getExtremeEigenvalues(M)
    # normalize eigenvalues of M between -1 to 1
    M_norm = normalizeEigen(M,smallest_ev,largest_ev)
    # generate random vectors (Gaussian distribution, could be wieh Rademacher)
    G = np.random.choice([-1, 1], size=(n, n_v))# np.random.randn(n,n_v) # 
    # normalize each column
    #norms = np.linalg.norm(G,axis=0,keepdims=True)
    # normalize columns
    #G = G/norms
    # Compute Chebyshev moments
    moments = np.zeros(p)
    T_prev = G
    T_curr = M_norm @ T_prev
    
    moments[0] = np.mean(np.sum(G * T_prev,axis = 0)) # T_{0}(A) = I
    moments[1] = np.mean(np.sum(G * T_curr,axis = 0)) # T_{1}(A) = A

    for k in range(2,p):
        T_next = 2*M_norm @ T_curr - T_prev
        moments[k] = np.mean(np.sum(G * T_next,axis = 0))
        T_prev,T_curr = T_curr,T_next
    # correct the weights of the moments
    weights = 2*np.ones(len(moments))
    weights[0] = 1
    weights = weights/(n*np.pi)
    # correct moments
    moments = weights*moments
    # use Jackson damping (smooth oscillations)
    x = np.linspace(-1,1,npoints)
    y = np.polynomial.chebyshev.chebval(x,moments)

    # scale the eigenvalues to their original support
    x = x*(largest_ev - smallest_ev)/2 + (largest_ev + smallest_ev)/2

    # return
    return (x,y)


def LanczosIteration(M,v,p,return_eigs = True):
    """
    Perform Lanczos iteration for eigenvalue approximation.

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.
    v : ndarray
        Column vector for initialization.
    p : int
        Number of Lanczos iterations.

    Returns
    -------
    return_eigs : ndarray or ndarray
        If True, returns the eigenvalues, otherwise returns the tridiagonal matrix.
    """

    n,_ = M.shape
    Q = np.zeros((n,p))
    alpha = np.zeros(p)
    beta = np.zeros(p - 1)

    Q[:,0] = v/np.linalg.norm(v)
    for j in range(p):
        w = M @ Q[:,j]
        alpha[j] = np.dot(Q[:,j],w)
        if j < p - 1:
            w -= alpha[j]*Q[:,j] + (beta[j-1]*Q[:,j-1] if j > 0 else 0)
            beta[j] = np.linalg.norm(w)
            if beta[j] == 0:
                break
            Q[:,j+1] = w/beta[j]
    # build the tridiagonal matrix using method for tridiagonal matrices
    if return_eigs:
        eig_vals,eig_vect = eigh_tridiagonal(alpha,beta)
        return (eig_vals,eig_vect[0,:]**2)
    else:
        return (alpha,beta)
    
# evaluate gaussian kernel
def evaluateGaussian(x,sigma):
    return (np.exp(-(x**2)/(2*(sigma**2)))/np.sqrt(2*np.pi)*sigma)



def LanczosSpectralDensity(M,p = 100,n_v = 50,sigma=0.1,npoints = 512):
    """
    Lanczos method for spectral density estimation.

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.
    p : int
        Number of Lanczos iterations.
    n_v : int
        Number of random vectors.
    npoints : int
        Number of discretization points.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    n,_ = M.shape
    # get smallest and largest eigenvector
    smallest_ev,largest_ev = getExtremeEigenvalues(M)
    print(f"{smallest_ev} {largest_ev}" )
    # obtain approximated eigenvalues
    approx_eigvals = [LanczosIteration(M,np.random.randn(n),p) for _ in range(n_v)]
    
    # get discretization points
    x = np.linspace(smallest_ev,largest_ev,npoints)
    Y = np.vstack([np.sum(np.vstack([sq_tau[idx]*evaluateGaussian(x - theta[idx],sigma) for idx in range(len(theta))]),axis=0) for theta,sq_tau in approx_eigvals])
    # Compute histogram density
    y = np.mean(Y,axis = 0)
    return x,y


####### HAYDOCK'S METHOD

# Compute the (1,1) entry of (zI - T_M)^{-1}
# using the recursion: 1/(z-alpha_{1}+\frac{beta_{2}^{2}}_{z - \alpha_{2} + ...})
def computeEntry(alpha,beta,z):
    entry_val = 1/(z - alpha[len(alpha) - 1])
    for idx in range(len(alpha) - 2,-1,-1):
        entry_val = 1/(z - alpha[idx] + (beta[idx]**2)*entry_val)
    return entry_val

def HaydocksSpectralDensity(M,p = 100,n_v = 50, eta = 0.1,npoints = 512):
    """
    Haydock's method for spectral density estimation.

    Parameters
    ----------
    M : ndarray
        Symmetric matrix.
    p : int
        Number of Lanczos iterations.
    n_v : int
        Number of random vectors.
    eta : float
        Lorentz smoothing factor.
    npoints : int
        Number of discretization points.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    n,_ = M.shape
    # get smallest and largest eigenvector
    smallest_ev,largest_ev = getExtremeEigenvalues(M)
    # get the discretization points
    x = np.linspace(smallest_ev,largest_ev,npoints)
    y = list() 
    # compute the tridiagonal matrices and the (1,1) entry of (zI - T)^{-1}
    (alpha,beta) = LanczosIteration(M,np.random.randn(n),p,return_eigs=False)
    entry_values = []
    z = x + 1j*eta
    for _ in range(n_v):
        (alpha,beta) = LanczosIteration(M,np.random.randn(n),p,return_eigs=False)
        entry_values.append(computeEntry(alpha,beta,z))
    # save the imaginary part
    y = np.mean(np.vstack(entry_values),axis = 0)
    y = [-val.imag for val in y]
    
    # return density
    return (x,y)


def MaxEntSpectralDensity(M,n_v = 50,l = 50,npoints = 512):
    """
    Maximum Entropy Method for spectral density estimation.

    Notes
    -----
    - This method only works for the normalized Laplacian matrix.
    - Generally applicable to matrices whose eigenvalue support is within [0,1].

    Parameters
    ----------
    M : ndarray
        Symmetric normalized Laplacian matrix.
    n_v : int
        Number of random vectors.
    l : int
        Largest moment to consider.
    npoints : int
        Number of discretization points.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """

    n,_ = M.shape
    M = M/2.0
    # Generate random vectors
    G = np.random.randn(n,n_v) # 
    # normalize each column
    norms = np.linalg.norm(G,axis=0,keepdims=True)
    # normalize columns
    G = G/norms
    # Compute Chebyshev moments
    moments = np.zeros(l)
    T_prev = G
    T_curr = M @ T_prev
    
    moments[0] = np.mean(np.sum(G * T_prev,axis = 0)) # T_{0}(A) = I
    moments[1] = np.mean(np.sum(G * T_curr,axis = 0)) # T_{1}(A) = A

    for k in range(2,l):
        T_next = 2*M @ T_curr - T_prev
        moments[k] = np.mean(np.sum(G * T_next,axis = 0))
        T_prev,T_curr = T_curr,T_next
    #
    # compute Chebyshev polynomial
    x = np.linspace(0,1,npoints)
    v = np.diff(x)
    v = np.append(v,0)
    n = len(moments)

    chebarray = np.zeros((n,npoints))
    for idx in range(0,n):
        q = (special.eval_chebyt(idx, x, out=None))
        chebarray[idx,:]= q

    # MaxEnt Algorithm
    # Entropic functional
    def s(alpha):
        j = 1 + np.dot(alpha,chebarray)
        q = np.exp(-j)
        u = (sum(q*v)) + np.dot(moments,alpha[:]);
        return u
    # Gradient of S wrt alpha
    def grad(alpha):
        u = []
        j = []
        j = 1+np.dot(alpha,chebarray)
        q = np.exp(-j)
        u = moments - np.dot(chebarray,(q*v))
        return np.asarray(u)
    # Hessian
    def hessian(alpha):
        j = []
        j = 1 + np.dot(alpha,chebarray)
        q = np.exp(-j)
        sd = np.array(chebarray)
        ass = np.einsum('j...,i...->ij...',sd,sd)
        ds = ass*q*v
        dss = np.sum(ds,axis=2)
        #Symmetrise and Add Jitter to improve Conditioning
        hdss = 0.5*(dss + dss.transpose())+1e-6*np.eye(len(alpha))
        return hdss
    
    # optimization
    # do trust-ncg with grad and hess
    res = minimize(s, x0 = np.ones(len(moments)), method = 'trust-ncg', jac=grad, hess=hessian, options={'gtol': 1e-5, 'disp': True, 'maxiter': None})
    # using the results
    alpha = res.x

    # use chebyt for chebyshev moments and x**i for power moments
    density_val = 1
    for i in range(0,len(alpha)):
        density_val = density_val + alpha[i]*((special.eval_chebyt(i, x, out=None)))

    p = 1/np.exp(density_val)
    MaxEntdistri = (1/np.exp(density_val))/(sum(p*v))

    # return results
    x = x*2
    return (x,MaxEntdistri)
