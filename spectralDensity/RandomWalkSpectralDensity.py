import numpy as np
import networkx as nx

import numpy as np
import cvxpy as cp

def solve_optimization_cvxpy(m, X):
    """
    Solves the optimization problem:
    min_{p} sum_{k=1}^{l} |m_k - sum_{j=1}^{t} p_k * x_j^k| 
    subject to sum_{k=1}^{l} p_k = 1 and p_k >= 0.
    
    Parameters:
        m : array-like, shape (l,)
            Target values.
        X : array-like, shape (l, t)
            Matrix containing x_j^k values.
    
    Returns:
        p_opt : array-like, shape (l,)
            Optimal values for p.
    """

    l, t = X.shape  # Number of p_k variables and summation terms

    # Decision variables
    p = cp.Variable(t, nonneg=True)  # p_k >= 0
    z = cp.Variable(l)  # Auxiliary variables for absolute value

    # Objective function: minimize sum of z_k
    objective = cp.Minimize(cp.sum(z))

    # Constraints
    constraints = [cp.sum(p) == 1]  # Probability constraint

    for k in range(l):
        constraints.append(z[k] >= m[k] - cp.sum(cp.multiply(X[k, :],p)))  # |m_k - sum p_{j} x_j^k|
        constraints.append(z[k] >= - (m[k] - cp.sum(cp.multiply(X[k, :],p))))  # Handling absolute value

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)  # SCS solver works well for L1-norm problems

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return p.value  # Return optimal p values
    else:
        raise ValueError("Optimization failed:", prob.status)

def ApproxSpectralMoment(G,l,s):
    """
    Approximate spectral moment method.

    Parameters
    ----------
    Graph : networkx.Graph
        Connected graph.
    l : int
        Moment to approximate.
    s : int
        Number of nodes to sample.

    Returns
    -------
    s : float
        Probability of finding a closed walk of length l.
    """
    all_nodes = G.nodes()
    count  = 0
    s_idx = s
    while s_idx != 0:
        u = np.random.choice(all_nodes)
        start = u
        for _ in range(l):
            neighbors = list(G.neighbors(u))
            u = np.random.choice(neighbors)
        if u == start:
            count = count + 1
        # next node
        s_idx = s_idx - 1
    #
    return count/s



def RandomWalkBasedSpectralDensity(G,l,s,npoints):
    """
    Estimate spectral density using a random-walk-based method.

    Notes
    -----
    - This method only works for the normalized Laplacian matrix.
    - Generally applicable to matrices whose eigenvalue support is within [0,1].

    Parameters
    ----------
    G : networkx.Graph
        Unweighted, undirected graph.
    l : int
        Largest moment to consider.
    s : int
        Number of nodes to sample.
    npoints : int
        Number of discretization points.

    Returns
    -------
    x : ndarray
        Discretized eigenvalue support with npoints elements.
    y : ndarray
        Estimated spectral density.
    """
    # discretization points
    x = np.linspace(0,2,npoints)
    # gamma = 2/(npoints+1)
    # compute l first moments
    moments = np.array([ApproxSpectralMoment(G,k,s) for k in range(1,l+1)])
    print(moments)
    # build power matrix
    X = np.vstack([x**k for k in range(1,l + 1)])
    # solve optimization problem to get the probabilities
    y = solve_optimization_cvxpy(moments,X)
    #
    return (x,y)