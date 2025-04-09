from spectralDensity import *

from utils import *

import networkx as nx
import numpy as np
## Load ca-CondMat graph
G = nx.read_edgelist("./data/pgp.txt", nodetype=int)
## Obtain the largest connected component
#* The reasor doing it is that some method require the graph to be connected.
# Get all connected components (each component is a set of nodes)
connected_components = list(nx.connected_components(G))

# Find the largest connected component
largest_component = max(connected_components, key=len)

# Optionally, get the subgraph of the largest component
H = G.subgraph(largest_component)
print(f"number of nodes: {H.number_of_nodes()}")
print(f"number of edges: {H.number_of_edges()}")
## Obtain its normalized Laplacian matrix
# obtain the Laplacian matrix
L = nx.normalized_laplacian_matrix(H)
ev = np.linalg.eigvalsh(L.toarray())
# Laplacian
H_lap = nx.from_scipy_sparse_array(L)

### Setting parameters for all the spectral density approximation methods

np.random.seed(0)
npoints = 512 # number of discretization points
eta = 0.1 # smooting factor, for the Message Passing and Haydock's method
sigma = 0.05 # smoothing factor for the Lanczos method
n_v = 100 # number of random vectors used for the numerical methods
l = 50 # largest moment to consider, Max Entropy and RandomWalk method
s =  1000 # RandomWalk method
p = 50 # chebyshev polynomial degree/number of Lanczos iteration
interval = (np.min(ev),np.max(ev)) # suport of the spectral density

### Run methods
x_real,y_real = exactSpectralDensity(eta,ev,npoints = npoints)
x_cheb,y_cheb = ChebyshevSpectralDensity(L,p = p,n_v = n_v,npoints = npoints) # CHebyshev
x_lanc,y_lanc = LanczosSpectralDensity(L,p = p,n_v = n_v,sigma = sigma,npoints=npoints) # Lanczo's
x_hayd,y_hayd = HaydocksSpectralDensity(L,p = p,n_v = n_v,eta = eta,npoints = npoints) # Hydock's
x_ment,y_ment = MaxEntSpectralDensity(L,n_v = n_v,l = l,npoints = npoints) # MaxEntropy
x_ltre,y_ltre = loopyMPSpectralDensity(H_lap,r = 0,eta = eta,interval = interval,npoints = npoints) # Message-Passing for Locally tree-like
x_dcor,y_dcor = degreecorrelatedSpectralDensity(H,eta = eta,mat = "nlap",interval=interval,npoints=npoints) # degree correlated method
x_deg,y_deg = degreeBasedSpectralDensity(H,eta = eta,mat = "nlap",interval=interval,npoints=npoints) # degree-based spectral density
x_shol,y_shol = loopyMPSpectralDensity(H_lap,r = 1,eta = eta,interval = interval,npoints = npoints) # Message-Passing for short loops
x_loop,y_loop = loopyMPSpectralDensity(H_lap,r = 2,eta = eta,interval = interval,npoints = npoints) # Message-Passing for loopy
x_rwal,y_rwal = RandomWalkBasedSpectralDensity(H,l = l,s = s,npoints = npoints) # Random Walk
### Save into pickle
# save into pickle file
import pickle

# List to save
my_list_names = ["Exact","Chebyshev","Lanczos","Haydock","maxEnt","loop MP r=0","degree-corr","degree-based","loop MP r=1","loop MP r=2","RW-based"]
my_list_x = [x_real,x_cheb,x_lanc,x_hayd,x_ment,x_ltre,x_dcor,x_deg,x_shol,x_loop,x_rwal]
my_list_y = [y_real,y_cheb,y_lanc,y_hayd,y_ment,y_ltre,y_dcor,y_deg,y_shol,y_loop,y_rwal]
results = {"names":my_list_names,"x":my_list_x,"y":my_list_y}

for idx,method_name in enumerate(my_list_names):
    dist = (my_list_x[idx] - my_list_x[0])**2 + (my_list_y[idx] - my_list_y[0])**2  
    print(f"{method_name} : {dist}",dist)

# Save to pickle file
with open('real_spectral_density.pkl', 'wb') as f:
    pickle.dump(results, f)


my_list_names = ["Exact","Chebyshev","Lanczos","Haydock","maxEnt","loop MP r=0","degree-corr","degree-based","loop MP r=1","loop MP r=2","RW-based"]
my_list_x = [x_real,x_cheb,x_lanc,x_hayd,x_ment,x_ltre,x_dcor,x_deg,x_shol,x_loop,x_rwal]
my_list_y = [y_real,y_cheb,y_lanc,y_hayd,y_ment,y_ltre,y_dcor,y_deg,y_shol,y_loop,y_rwal]

### Plot figure
import matplotlib.pyplot as plt
from matplotlib import cm

# Define line styles and markers to cycle through
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', '*', 'x', 'v', '<', '>', 'P']
colors = cm.get_cmap('tab20', 11)  # Colormap with 10 distinct colors

plt.figure(figsize=(10, 8))
for i in range(len(my_list_x)):
    x = my_list_x[i]
    y = my_list_y[i]
    linewidth = 2
    if i == 0:
        linewidth = 3
    plt.plot(
        x, y,
        label=my_list_names[i],
        linestyle=line_styles[i % len(line_styles)],
        marker=markers[i % len(markers)],
        markevery=64,  # place markers every 10 points
        color=colors(i),
        linewidth=linewidth
    )
plt.xlabel(r"$\mathbf{\alpha}$", fontsize=14)
plt.ylabel(r"$\mathbf{\rho(\lambda)}$", fontsize=14)
plt.legend(fontsize=12,prop={'weight': 'bold'})
#plt.title("Spectral Density")
# Get current axes
ax = plt.gca()

# Make tick labels bold
ax.tick_params(axis='both', which='both', labelsize=12, width=2)
#
# Save the figure as a PDF
plt.savefig("real_world.pdf", format='pdf')
# show
plt.show()
