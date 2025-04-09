import numpy as np
### compute spectral density given eta (cauchy/Lorentz distribution)
# compute spectral density with broadening parameter eta, from
# the true eigenvalues (ev)
def rho(x, eta, ev):
    z = x+(0+1j)*eta
    return np.imag(-np.mean(1/(z-ev))/np.pi)

# get exact spectral density
# Parameters:
# eta: broadening parameter
# ev: eigenvalues
# npoints: discretization points
# 
# Return:
# (x,y): components of the spectral density
def exactSpectralDensity(eta,ev,npoints = 512):
    x = np.linspace(np.min(ev),np.max(ev),npoints)
    y = np.array([rho(x_val,eta,ev) for x_val in x])

    return (x,y)