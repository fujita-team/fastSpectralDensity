import numpy as np
import networkx as nx
from .MPutils import GraphDecomposition

#Update the cavity nodes variances until convergence is attained

def MPCavityGraph(Gdecomp,z,tol=1e-8,max_iter = 5000):
    while(max_iter != 0):
        error = Gdecomp.update_cavity_messages(z)
        if(error < tol):
            break
        max_iter = max_iter - 1


def loopyMPSpectralDensity(G,r,eta = 0.1,interval = (None,None),npoints = 512,tol=1e-6):
    """
    Perform message passing using the cavity method, including all message passing variants.

    Parameters
    ----------
    G : networkx.Graph
        Weighted graph. Can also represent the Laplacian, normalized Laplacian, and other matrices.
    r : int
        Largest primitive cycle length to consider. 
        - r = 0: Locally tree-like structure.
        - Fast version only works with r <= 3.
    eta : float
        Smoothing factor (Cauchy/Lorentz distribution).
    range : tuple of float
        Interval (from, to) that includes all eigenvalues of G.
    npoints : int
        Number of discretization points.
    tol : float
        Error tolerance.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    # obtain teh graph decomposition
    Gdecomp = GraphDecomposition(G,r=r)
    # get the sum of the edges weight of each node. Will be used as an approximation for the eigenvalue support
    if interval[0] == None:
        weighted_degrees = [w for _,w in G.degree(weight="weight")]
        max_wdeg = np.max(weighted_degrees)
        smallest_ev = -max_wdeg
        largest_ev = max_wdeg
    else:
        smallest_ev = interval[0]
        largest_ev = interval[1]
    ##
    x = np.linspace(smallest_ev,largest_ev,npoints)
    y = list()
    #
    z = x - eta*1j
    for idx_point in range(len(z)):
        Gdecomp.initialize_cavity_messages()
        # update values of the cavity variances until convergence
        MPCavityGraph(Gdecomp,z[idx_point],tol)
        # compute the variance of all nodes
        all_var = Gdecomp.compute_nodes_variance(z[idx_point])
        y.append(np.imag(np.mean(1.0/(z[idx_point] - all_var))/np.pi))
        if idx_point % 200 == 0:
            print(f"Computing value at {idx_point}")
    #
    return (x,y)

######################

def s(d_u,d_v,z,mat):
    if mat == 'adj':
        return (1/(z**2))
    elif mat == 'lap':
        return (1/((z-d_u)*(z-d_v)))
    elif mat == "nlap":
        return (1/(d_u*d_v*((z - 1)**2)))

def H(a,mat):
    if mat == 'adj':
        return 0
    elif mat == 'lap':
        return a
    elif mat == 'nlap':
        return 1
    

def MPdegreecorrelated(P,udegree,z,mat = "adj",max_iter = 5000,tol = 1e-8):
    """
    Solve message passing equations for degree-correlated matrices.

    Parameters
    ----------
    P : ndarray
        Degree-degree correlation matrix.
    udegree : array-like
        Sequence of unique degrees.
    deg_distb : array-like
        Degree probability distribution.
    z : complex
        Complex point where the evaluation is performed.
    mat : str
        Indicates the matrix type for spectral density computation.
        Options:
        - 'adj' : Adjacency matrix
        - 'lap' : Laplacian matrix
        - 'nlap': Normalized Laplacian matrix
    max_iter : int
        Maximum number of iterations.
    tol : float
        Error tolerance.

    Returns
    -------
    R : ndarray
        Matrix with the converged message passing values.
    """
    nudegree = len(udegree)
    R = np.zeros((nudegree,nudegree),dtype=np.complex128)
    # R[i,j] = h^{i <- j}
    while max_iter != 0:
        new_R = np.zeros((nudegree,nudegree),dtype=np.complex128)
        for a in range(nudegree):
            for b in range(nudegree):
                # compute the value of h^{b <- a}
                K = udegree[b]*np.ones(nudegree)
                K = K*P[b,:]
                K[a] = max(K[a] - 1,0)
                new_R[b,a] = s(udegree[a],udegree[b],z,mat)/(1.0 - (udegree[b] - 1)*np.sum(K*R[:,b]))#np.sum(P[b,:]*R[:,b]))
        #
        error = np.sum(np.abs(new_R - R))
        R = new_R
        if error < tol*(nudegree**2):
            break
    # return value
    return (R)


def degreecorrelatedSpectralDensity(G,eta = 0.1,mat = "adj",interval = (None,None),npoints = 512,tol = 1e-8):
    """
    Perform message passing for degree-correlated matrices.

    Parameters
    ----------
    G : networkx.Graph
        Unweighted graph without self-loops. Can also represent the Laplacian, 
        normalized Laplacian, and other matrices.
    eta : float
        Smoothing factor (Cauchy/Lorentz distribution).
    mat : str
        Indicates the matrix type for spectral density computation.
        Options:
        - 'adj' : Adjacency matrix
        - 'lap' : Laplacian matrix
        - 'nlap': Normalized Laplacian matrix
    range : tuple of float
        Interval (from, to) that includes all eigenvalues of G.
    npoints : int
        Number of discretization points.
    tol : float
        Error tolerance.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    n = G.number_of_nodes()
    # get edge list (does not have self loops)
    edge_list = list(G.edges())
    # get the degree of all nodes
    node_degree = [G.degree[node] for node in G.nodes()]
    # build the degree frequency
    udegree = list(set(node_degree))
    idx_udegree = {deg:idx for idx,deg in enumerate(udegree)}
    deg_distb = np.zeros(len(udegree))
    for deg in node_degree:
        deg_distb[idx_udegree[deg]] = deg_distb[idx_udegree[deg]] + 1
    # now normalize
    deg_distb = deg_distb/n
    # compute the
    ## build the deg correlation matrix
    P = np.zeros((len(deg_distb),len(deg_distb)))  
    for u,v in edge_list:
        d_u,d_v = G.degree[u],G.degree[v]
        P[idx_udegree[d_u],idx_udegree[d_v]] += 1
        P[idx_udegree[d_v],idx_udegree[d_u]] += 1 
    #normalize by column
    P = P/P.sum(axis=0,keepdims=1)

    ##### COMPUTE THE SPECTRAL DENSITY    
    # get the sum of the edges weight of each node. Will be used as an approximation for the eigenvalue support
    udegree = np.array(udegree)
    if interval[0] == None:
        max_deg = np.max(udegree)
        if mat == 'adj':
            smallest_ev,largest_ev = -max_deg,max_deg
        elif mat == 'lap':
            smallest_ev,largest_ev = 0,2*max_deg
        elif mat == 'nlap':
            smallest_ev,largest_ev = 0,2
    else:
        smallest_ev = interval[0]
        largest_ev = interval[1]
    ##
    x = np.linspace(smallest_ev,largest_ev,npoints)
    y = list()
    # evaluate spectral density for each discretized point
    z = x - eta*1j
    for idx_point in range(len(z)):
        # compute message passing until convergence
        R = MPdegreecorrelated(P,udegree,z[idx_point],mat = mat,tol=tol)
        # compute the spectral density
        g = list()
        for id_deg in range(len(udegree)):
            val_l = (1/(z[idx_point] - H(udegree[id_deg],mat)))
            val_r = deg_distb[id_deg]/(1 - udegree[id_deg]*np.sum(P[id_deg,:]*R[:,id_deg]))
            g.append(val_l*val_r)
        #
        y.append(np.imag(np.sum(g))/np.pi)
    # return values
    return (x,y)


###################### DEGREE-BASED METHOD

# global mean sumatory terms
# returns terms that will be used to compute the cavity mean
def sumTerms(z,udeg_mo,mat = 'adj'):
    if mat == 'adj':
        return (1/(z**2),np.ones(len(udeg_mo)))
    elif mat == 'lap':
        return (1+0.j,z - udeg_mo - 1)
    elif mat == 'nlap':
        return (1/((z-1)**2),udeg_mo + 1)
    

def MPcavityMean(udeg,edeg_distb,z,mat = 'adj',max_iter = 5000,tol = 1e-8):
    """
    Compute the cavity mean h(z).

    Parameters
    ----------
    udeg : array-like
        Unique degrees.
    edeg_distb : array-like
        Excess degree distribution.
    z : complex
        Complex point where the evaluation is performed.
    mat : str
        Indicates the matrix type for spectral density computation.
        Options:
        - 'adj' : Adjacency matrix
        - 'lap' : Laplacian matrix
        - 'nlap': Normalized Laplacian matrix
    max_iter : int
        Maximum number of iterations.
    tol : float
        Error tolerance.

    Returns
    -------
    h_z : complex
        Global cavity mean.
    """
    udeg_mo = udeg - 1
    out_sum,w_sum = sumTerms(z,udeg_mo,mat = mat)
    # compute global mean: h(z)
    h_z = 0+1.j
    while max_iter != 0:
        h_z_new = out_sum*np.sum(edeg_distb/(w_sum - udeg_mo*h_z))
        error = np.abs(h_z - h_z_new)
        h_z = h_z_new
        if error < tol:
            break
        max_iter = max_iter - 1
    # return global mean
    return h_z

# auxiliar function to compute the final density given the cavity mean and degree distribution
def computeDensity(h_z,z,udegree,deg_distb,mat = 'adj'):
    if mat == 'adj':
        return np.imag((1/z)*np.sum((deg_distb/(1 - udegree*h_z))))/np.pi
    elif mat == 'lap':
        return np.imag(np.sum(deg_distb/((z - udegree) - udegree*h_z)))/np.pi
    elif mat == "nlap":
        return np.imag((1/(z-1))*(1/(1-h_z)))/np.pi

def degreeBasedSpectralDensity(G,eta = 0.1,mat = "adj",interval = (None,None),npoints = 512,tol = 1e-8):
    """
    Compute the spectral density given a complex point and the global cavity mean.

    Parameters:
    -----------
    h_z : complex
        Global cavity mean.
    z : complex
        Complex point.
    udegree : array-like
        Unique degree sequence.
    deg_distb : array-like
        Degree distribution.
    mat : str
        Indicates the matrix type for spectral density computation.
        Options:
        - 'adj' : Adjacency matrix
        - 'lap' : Laplacian matrix
        - 'nlap': Normalized Laplacian matrix

    Returns:
    --------
    float
        Spectral density at the given complex point.
    """

    n = G.number_of_nodes()
    # get the degree of all nodes
    node_degree = [G.degree[node] for node in G.nodes()]
    # build the degree frequency
    udegree = np.array(list(set(node_degree)))
    idx_udegree = {deg:idx for idx,deg in enumerate(udegree)}
    deg_distb = np.zeros(len(udegree))
    for deg in node_degree:
        deg_distb[idx_udegree[deg]] = deg_distb[idx_udegree[deg]] + 1
    # now normalize
    deg_distb = deg_distb/n
    edeg_distb = deg_distb*udegree
    edeg_distb = edeg_distb/np.sum(edeg_distb)
    ##### COMPUTE THE SPECTRAL DENSITY    
    # get the sum of the edges weight of each node. Will be used as an approximation for the eigenvalue support
    if interval[0] == None:
        max_deg = np.max(udegree)
        if mat == 'adj':
            smallest_ev,largest_ev = -max_deg,max_deg
        elif mat == 'lap':
            smallest_ev,largest_ev = 0,2*max_deg
        elif mat == 'nlap':
            smallest_ev,largest_ev = 0,2
    else:
        smallest_ev = interval[0]
        largest_ev = interval[1]
    ##
    x = np.linspace(smallest_ev,largest_ev,npoints)
    y = list()
    # evaluate spectral density for each discretized point
    z = x - eta*1j
    for idx_point in range(len(z)):
        # compute message passing until convergence
        h_z = MPcavityMean(udegree,edeg_distb,z[idx_point],mat=mat,tol=tol)
        # compute the spectral density
        y.append(computeDensity(h_z,z[idx_point],udegree,deg_distb,mat = mat))
    # return values
    return (x,y)