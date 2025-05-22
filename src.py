import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Constants
m = 0.001
hbar = 1.0
L = 10.0
N = 1000
dr = L / (N-1) 
r = np.linspace(0, L, N)
V = np.zeros(N)
dt = 0.001  
T_steps = 1000 

# Potential
V = np.zeros(N)
V[int(N/4):int(3*N/4)] = 1.0

# Tridiagonal matrix for the kinetic term
diagonal = -2.0 * np.ones(N)
off_diagonal = 1.0 * np.ones(N-1)
kinetic = (-hbar**2 / (2 * m * dr**2)) * sparse.diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1])

# Potential term
potential = sparse.diags(V)

# Total Hamiltonian
H = kinetic + potential

# Matrices for the Crank-Nicolson method
I = sparse.identity(N)
A = (I + 1j * dt / (2 * hbar) * H).tocsc()
B = (I - 1j * dt / (2 * hbar) * H).tocsc()

# Initial wave function --> Gaussian
psi = np.exp(-((r - L/2)**2) / 2)
psi = psi / np.linalg.norm(psi) 

# Time evolution
plt.figure(figsize=(10, 6))
for t in range(T_steps):
    psi = spsolve(A, B.dot(psi))
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dr)

    if t % 100 == 0:
        plt.plot(r, np.abs(psi)**2, label=f't={t*dt:.2f}')

plt.title("Time Evolution of the Wave Function")
plt.xlabel("x")
plt.ylabel("Probability Density |\u03C8(x)|^2")
plt.legend()
plt.grid(True)
plt.show()