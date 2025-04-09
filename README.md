# Spectral Density Estimation for Large Sparse Graphs

This repository provides a collection of methods for estimating the **spectral density** of large sparse graphs. These methods cover a wide range of approximation techniques, including polynomial expansions, random walks, message-passing, and entropy maximization.

The spectral density (or density of states) provides key insights into the structure and dynamics of networks, especially in physics-inspired models, graph signal processing, and complex systems analysis.

## 📦 Methods Included

### 1. **Exact Spectral Density**
```python
x_real, y_real = exactSpectralDensity(eta, ev, npoints=npoints)
```
Computes the true spectral density by directly using the eigenvalues `ev` and a broadening parameter `eta`.

---

### 2. **Chebyshev Polynomial Expansion**
```python
x_cheb, y_cheb = ChebyshevSpectralDensity(L, p=p, n_v=n_v, npoints=npoints)
```
Approximates the spectral density using Chebyshev polynomials and stochastic trace estimation.

---

### 3. **Haydock's Recursive Method**
```python
x_hayd, y_hayd = HaydocksSpectralDensity(L, p=p, n_v=n_v, eta=eta, npoints=npoints)
```
Recursive method based on continued fraction expansion (popular in physics literature).

---

### 4. **Lanczos Algorithm**
```python
x_lanc, y_lanc = LanczosSpectralDensity(L, p=p, n_v=n_v, sigma=sigma, npoints=npoints)
```
Lanczos-based method that builds a tridiagonal approximation of the matrix spectrum.

---

### 5. **Maximum Entropy Method**
```python
x_ment, y_ment = MaxEntSpectralDensity(L, n_v=n_v, l=l, npoints=npoints)
```
Uses the principle of maximum entropy to estimate the density from limited spectral moments.

---

### 6. **Message-Passing Approximations (Loopy Belief Propagation)**
```python
x_ltre, y_ltre = loopyMPSpectralDensity(H, r=0, eta=eta, interval=interval, npoints=npoints) # Locally tree-like
x_shol, y_shol = loopyMPSpectralDensity(H, r=1, eta=eta, interval=interval, npoints=npoints) # Short loops
x_loop, y_loop = loopyMPSpectralDensity(H, r=2, eta=eta, interval=interval, npoints=npoints) # Loopy structure
```
Approximates the spectral density using belief-propagation-based techniques, adapting to various levels of loopiness (`r=0`, `1`, or `2`).

---

### 7. **Random Walk-Based Estimation**
```python
x_rwal, y_rwal = RandomWalkBasedSpectralDensity(G, l=l, s=s, npoints=npoints)
```
Uses short random walks to probe the spectrum via return probabilities.

---

### 8. **Degree-Correlated Approximation**
```python
x_dcor, y_dcor = degreecorrelatedSpectralDensity(G, eta=eta, mat="nlap", interval=interval, npoints=npoints)
```
Estimates the spectral density incorporating degree-degree correlations of the graph.

---

### 9. **Degree-Based Approximation**
```python
x_deg, y_deg = degreeBasedSpectralDensity(G, eta=eta, mat="nlap", interval=interval, npoints=npoints)
```
Simpler approximation that only depends on the degree distribution.

---

## 📘 Requirements

- Python 3.7+
- `numpy`, `scipy`, `networkx`, `matplotlib` (for optional plotting)
- Custom modules as required (some methods may depend on specialized libraries for numerical routines)

Install dependencies via:

```bash
pip install -r requirements.txt
```

## 📈 Example Usage

```python
import networkx as nx
from yourmodule import ChebyshevSpectralDensity

G = nx.erdos_renyi_graph(n=1000, p=0.01)
L = nx.normalized_laplacian_matrix(G)

x, y = ChebyshevSpectralDensity(L, p=100, n_v=10, npoints=200)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.title("Chebyshev Spectral Density")
plt.show()
```

## 🔍 Comparison

All methods aim to estimate the spectral density but differ in:
- **Accuracy** vs **Computational cost**
- **Assumptions** (e.g., local tree-likeness, degree correlations)
- **Memory requirements** (some avoid storing full matrices)

Use `exactSpectralDensity` only for small graphs (for validation), and the others based on the size/structure of your graph.

## 📄 License

MIT License

## 🤝 Contributing

Feel free to submit pull requests or report issues! Contributions are welcome for additional methods, performance improvements, or better documentation.

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
