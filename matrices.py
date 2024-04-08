import numpy as np
from itertools import product
import time


def Hamiltonian(L, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, w=0, disorder=[1, 1, 1]):
    """
    Generate the Hamiltonian for a 3D lattice.

    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    k_z (float): z-direction momentum.
    w (float): Disorder strength.
    disorder (list): Disorder type.
    Fourier (bool): Fourier transform.

    Returns:
    Operator: Hamiltonian for the 3D lattice.
    """

    assert ''.join([char for char, flag in zip("xyz", disorder) if flag == 0]) in PBC
    assert sum(disorder) <= 1

    starttime = time.time()
    disorder_array = generate_disorder(L, w, disorder)
    H = np.zeros([L[1]*L[2], 2*L[0], 2*L[0]], dtype=complex)

    x_op = np.array([[0, 1], [1, 0]])
    y_op = np.array([[0, -1j], [1j, 0]])
    z_op = np.array([[1, 0], [0, -1]])

    for x, y, z in product(range(L[0]), range(L[1]), range(L[2])):
        H[y+z*L[1], 2*x:2*(x+1), 2*x:2*(x+1)] = t*np.sin(2*np.pi*x/L[0])*x_op + \
            t*np.sin(2*np.pi*y/L[1])*y_op + \
            (t_z*np.cos(2*np.pi*z/L[2])-m_z+t_0 *
            (2-np.cos(2*np.pi*x/L[0])-np.cos(2*np.pi*y/L[1])))*z_op
            
    x_matrix = np.zeros([L[1], L[1]], dtype=complex)

    for i, j in product(range(L[1]), range(L[1])):
        x_matrix[i, j] = np.exp(-2*np.pi*1j*i*j/L[1])/np.sqrt(L[1])

    t_matrix = np.kron(x_matrix, np.eye(2))

    mat = np.zeros([2*L[0], 2*L[0]], dtype=complex)
    if PBC != "xyz":
        for s in range(2):
            mat[s + 2*(L[0] - 1), (s + 1) %
            2] = t/2*1j
            mat[s + 2*(L[0] - 1), s
            ] = t_0/2*(-1)**s
        mat = mat + mat.conj().T
        
        H_op = np.array([t_matrix.conj().T @ H[i] @ t_matrix + disorder_array + mat for i in range(L[1]*L[2])])
            
    print("Hamiltonian built in ", time.time()-starttime)

    return H_op


def generate_C(H, L, axis, FT="", get_spectrum=False, fill=512):
    """
    Generate the entanglement matrix for a 3D lattice.

    Parameters:
    H (Operator): Hamiltonian for the 3D lattice.
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    subset (numpy.ndarray): Subset of lattice sites for entanglement cut.
    FT (str): Fourier transform type.

    Returns:
    numpy.ndarray: Correlation matrix for the 3D lattice.
    """

    assert len(FT) <= 2 and axis not in FT

    assist = {"x": 0, "y": 1, "z": 2}
    Ln = L.copy()
    Ln[assist[axis]] = Ln[assist[axis]]//2

    assert len(FT) == 1
    starttime = time.time()
    E, V = np.linalg.eigh(H)
    print("Diagonalization time:", time.time()-starttime)
    sorted_indices = np.argsort(E.flatten())
    lower_half = sorted_indices[:sorted_indices.shape[0]*fill//1024]
    # Optimized creation of new_V
    starttime = time.time()
    indices = lower_half % (2*L[0])
    grouped_indices = [indices[lower_half // (2*L[0]) == i] for i in range(L[1]*L[2])]
    new_V = [V[i][:, idx][:L[0], :] for i, idx in enumerate(grouped_indices)]
    print("New V time:", time.time()-starttime)

    # Optimized calculation of C
    print("Calculating C...")
    C = np.array([v.conj() @ v.T for v in new_V])
    if get_spectrum:
        return [C, E]
    return C


def generate_disorder(L, w, directions):
    """
    Generate the disorder array for a 3D lattice.

    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    w (float): Disorder strength.
    directions (list): Disorder type.

    Returns:
    numpy.ndarray: Disorder array for the 3D lattice.
    """
    s = sum(directions)
    assert len(directions) == 3 and s % 2 == 1 or s == 0

    assert s == 1
    assert directions[0] == 1
    rand = np.random.uniform(-w/2, w/2, size=L[0])
    disorder_array = np.diag(rand)
    disorder_array = np.kron(disorder_array, np.eye(2))
    return disorder_array


