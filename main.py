import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool
import datetime
from matrices import *
import time
import h5py
import os
import tqdm as tqdm
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment
import seaborn as sns

cwd = os.getcwd()
print(cwd)


# Set the 'axes.grid' parameter to True globally
plt.rcParams['axes.grid'] = True


def calc_entropy(arr, diagonalize=False, fast=False):
    """
    Calculate the entanglement entropy for a 3D lattice.

    Parameters:
    arr (numpy.ndarray): Input array.
    diagonalize (bool): Diagonalize the input array.
    fast (bool): Use the fast method for the entanglement spectrum.


    Returns:
    numpy.ndarray: Entanglement entropies.
    """
    startentropy = time.time()
    
    if fast:
        print(arr.shape, arr.ndim)
        if arr.ndim == 3:
            entropies = np.sum(-arr*np.log(arr) - (1-arr)*np.log(1-arr), axis=(1, 2))
        elif arr.ndim == 2:
            arr = arr[(arr>0) & (arr<1)]
            entropies = np.sum(-arr*np.log(arr) - (1-arr)*np.log(1-arr))
        return entropies

    if diagonalize:
        eigvals = np.linalg.eigvalsh(arr)

    else:
        eigvals = arr

    if np.ndim(eigvals) == 1:
        entropies = -sum([(1-f)*np.log(1-f) + f*np.log(f)
                          for f in eigvals if 0 < f < 1])
        
    elif np.ndim(eigvals) == 2:
        entropies = np.zeros(eigvals.shape[0])

        for i, a in enumerate(eigvals):
            entropies[i] = -sum([(1-f)*np.log(1-f) + f*np.log(f)
                                for f in a if 0 < f < 1])
            
    else:
        raise ValueError("Invalid dimension for C_input.")
    
    print("Time taken for entropy:", time.time() - startentropy)
    
    return entropies


def calc_disordered_spectra(L, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, FT="y", axis="x", shift=False, w=0, disorder=[1, 1, 1], N_max=10, history=True, num_bins=20, fast=False, SAVE=False):
    """
    Plot the physical and entanglement spectra for a 3D lattice with disorder.

    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    axis (str): Axis for entanglement cut.
    shift (bool): Shift the physical spectrum.
    w (float): Disorder strength.
    disorder (list): Disorder directions.
    N_max (int): Number of iterations.
    history (bool): Plot the history of the entanglement spectrum.
    num_bins (int): Number of bins for the density of states.

    Returns:
    None
    """

    w_array = np.atleast_1d(w)
    fig, axes = plt.subplots(3, 2, figsize=(8, 6))

    if history:
        for _, w in enumerate(w_array):
            print("Calculating spectra for w = " + str(w) + "...")
            if fast:
                w, eigvals, spectrum, DOS, entropies, eigvecs = compute_eigenvalues(w, L, PBC, t_z, t, t_0,
                                                                       m_z, N_max, axis, FT, num_bins, disorder, fast=fast)
            else:
                w, entanglement_spectrum, physical_spectrum, DOS, entropies = compute_eigenvalues(w, L, PBC, t_z, t, t_0,
                                                                       m_z, N_max, axis, FT, num_bins, disorder)
            axes[1, 1].plot(entropies, 'o',
                            label='Entanglement Entropies of all iterations')
            axes[2, 0].plot(DOS[0], DOS[1], '-',
                            label='Density of States of last iteration')
            axes[2, 1].plot(w, DOS[1][DOS[1].shape[0]//2], 'o',
                            label='Density of States at E = 0 of last iteration')
        
        if fast:
            physical_spectrum, entanglement_spectrum, entvecs = np.zeros((L[2], 2*L[0]*L[1])), np.zeros((L[2], L[0]*L[1])), np.zeros((L[2], L[0]*L[1], L[0]*L[1]), dtype=complex)
            for i in range(eigvals.shape[0]):
                ind = (L[0]*(i % L[1]))
                physical_spectrum[i//L[1], (2*ind):(2*(ind + L[0]))] = spectrum[i, :]
                entanglement_spectrum[i//L[1], ind:(ind + L[0])] = eigvals[i, :]
                entvecs[i//L[1], ind:(ind+L[0]), ind:(ind+L[0])] = eigvecs[i, :]
            
        plot_with_shift(physical_spectrum, axes[0, 0], shift=shift,
                        label='Physical Spectrum of last iteration', symbol='.')
        plot_with_shift(entanglement_spectrum, axes[0, 1], shift=shift,
                        label='Entanglement Spectrum of last iteration', symbol='.')
        if fast:
            rows, cols = np.where(np.abs(entanglement_spectrum - 0.5) < 499e-3) 
            indexpairs = list(zip(rows, cols))
            for i in range(10):
                random_index_pair = indexpairs[np.random.choice(len(indexpairs))]
                sorted_vecs = entvecs[np.ix_(np.atleast_1d(random_index_pair[0]), np.arange(entvecs.shape[1]), np.atleast_1d(random_index_pair[1]))]
                Ln = [L[0]//2, L[1], L[2]]
                plot_localization(Ln, sorted_vecs, axis="x", axes=axes[1, 0], energies = entanglement_spectrum[random_index_pair[0], random_index_pair[1]])
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('Localization of 1/2 eigenstates')
            axes[1, 0].legend()
            if SAVE:
                current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data\\{current_datetime}_results_N={L[0]}_w={w:.2f}.h5"
                with h5py.File(filename, 'w') as file:
                    file.create_dataset("w", data=w)
                    file.create_dataset("avg_ES", data=eigvals)
                    file.create_dataset("last_PS", data=spectrum)
                    file.create_dataset("DOS", data=DOS)
                    file.create_dataset("entropies", data=entropies)
                    file.create_dataset("eigvecs", data=eigvecs)
        
        else:
            plot_with_shift(np.log((1-eigvals)/eigvals), axes[1, 0], shift=shift,   
                        label='Entanglement Energy of last iteration', symbol='-')
            axes[1, 0].set_xlabel(f'k_{FT}')
            axes[1, 0].set_ylabel('Entanglement Energy')

        axes[0, 0].set_xlabel(f'k_{FT}')
        axes[0, 0].set_ylabel('Physical Spectrum')
        axes[0, 0].set_ylim([-0.8, 0.8])

        axes[0, 1].set_xlabel(f'k_{FT}')
        axes[0, 1].set_ylabel('Entanglement Spectrum')

        axes[1, 1].set_xlabel('index')
        axes[1, 1].set_ylabel('Entanglement Entropy')

        axes[2, 0].set_xlabel('Energy')
        axes[2, 0].set_ylabel('Density of States')

        axes[2, 1].set_xlabel('w')
        axes[2, 1].set_ylabel('Density of States at E = 0')

        if shift:
            tick_positions = [-np.pi, 0, np.pi]
            tick_labels = ['$-\pi$', '0', '$\pi$']

        else:
            tick_positions = [0, np.pi, 2*np.pi]
            tick_labels = ['0', '$\pi$', '$2\pi$']

        for ax in axes.flat[:2]:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

        fig.suptitle('Disordered Spectra for w = ' + str(w_array))
        plt.show()

    else:
        for w in w_array:
            C = np.zeros([np.prod(L), np.prod(L)], dtype=complex)
            for i in range(N_max):
                print(f"Iteration {i+1} of {N_max}")
                H = Hamiltonian(L, PBC, t_z, t, t_0,
                                m_z, w, disorder=disorder)
                C = ((i)*C + generate_C(H, L, axis=axis, FT=FT))/(i+1)
            eigvals = C.subdiag(FT).real
            spectrum = H.subdiag(FT).real
            num_bins = 20
            DOS = get_DOS(spectrum.flatten(), num_bins=num_bins)
            plot_with_shift(spectrum, axes[0, 0], shift=shift,
                            label='Physical Spectrum', symbol='-')
            plot_with_shift(eigvals, axes[0, 1], shift=shift,
                            label='Entanglement Spectrum', symbol='.')
            # axins.plot(eigvals, 'o', label='w='+str(w))
            plot_with_shift(np.log((1-eigvals)/eigvals), axes[1, 0], shift=shift,
                            label='Entanglement Energy', symbol='-')
            axes[1, 1].plot(DOS[0], DOS[1], '-',
                            label='Density of States')

        axes[0, 0].set_xlabel(f'k_{FT}')
        axes[0, 0].set_ylabel('Physical Spectrum')

        axes[0, 1].set_xlabel(f'k_{FT}')
        axes[0, 1].set_ylabel('Entanglement Spectrum')

        axes[1, 0].set_xlabel(f'k_{FT}')
        axes[1, 0].set_ylabel('Entanglement Energy')

        axes[1, 1].set_xlabel('Energy')
        axes[1, 1].set_ylabel('Density of States')

        if shift:
            tick_positions = [-np.pi, 0, np.pi]
            tick_labels = ['$-\pi$', '0', '$\pi$']
        else:
            tick_positions = [0, np.pi, 2*np.pi]
            tick_labels = ['0', '$\pi$', '$2\pi$']

        for ax in axes.flat[:3]:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

        fig.suptitle('Disordered Spectra for w = ' + str(w_array))
        plt.show()


def compute_eigenvalues(w, L, PBC, t_z, t, t_0, m_z, N_max, axis, FT, num_bins, disorder=[1, 1, 1]):
    """
    Compute the physical and entanglement spectra for a 3D lattice with disorder.

    Parameters:
    w (float): Disorder strength.
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    N_max (int): Number of iterations.
    axis (str): Axis for entanglement cut.
    FT (str): Fourier transform type.
    num_bins (int): Number of bins for the density of states.
    disorder (list): Disorder directions.

    Returns:
    numpy.ndarray: Correlation matrix for the 3D lattice.
    """

    totalstart = time.time()
    assert len(FT) <= 1

    print("Calculating spectra for w = " + str(w) + "...")

    DOS_array = np.zeros([N_max, num_bins-1])
    H = Hamiltonian(L, PBC, t_z, t, t_0, m_z, w,
                disorder=disorder)
    [C_roll, spectrum] = generate_C(H, L, axis=axis, FT=FT, get_spectrum=True)
    for i in range(N_max-1):
        print(f"Iteration {i+2} of {N_max}")
        DOS_array[i] = get_DOS(spectrum.flatten(), num_bins=num_bins)[1]
        H = Hamiltonian(L, PBC, t_z, t, t_0, m_z, w,
                disorder=disorder)
        [C_new, spectrum] = generate_C(H, L, axis=axis, FT=FT, get_spectrum=True)
        C_roll = ((i+1) * C_roll +  C_new)/ (i + 2)
    eigvals, eigvecs = np.linalg.eigh(C_roll)
    print(eigvals.shape, eigvecs.shape)
    entropies = calc_entropy(eigvals, diagonalize=False, fast=True)
    DOS = get_DOS(spectrum.flatten(), num_bins=num_bins)
    DOS_array[-1] = DOS[1]
    print("Time taken:", time.time() - totalstart)
    return w, eigvals, spectrum, [DOS[0], np.average(DOS_array, axis=0)], entropies, eigvecs


def get_DOS(spectrum, num_bins=100, plot=False, combine=False, with_std=False):
    """
    Compute the density of states for a 3D lattice.

    Parameters:
    spectrum (numpy.ndarray): Spectrum for the 3D lattice.
    num_bins (int): Number of bins for the density of states.
    plot (bool): Plot the density of states.
    combine (bool): Combine the density of states.
    with_std (bool): Include the standard deviation in the plot.

    Returns:
    numpy.ndarray: Density of states for the 3D lattice.
    """

    spectrum = np.atleast_2d(spectrum)
    l = spectrum.shape[0]
    if l > 1:
        DOS = np.zeros((l, 2, num_bins-1))

    for i, s in enumerate(spectrum):
        bins = np.linspace(0, 1, num_bins)
        bins_center = (bins[1:] + bins[:-1])/2
        hist, bin_edges = np.histogram(s, bins=bins, density=False)

        if l == 1:
            DOS = np.array([bins_center, hist])

        else:
            DOS[i] = np.array([bins_center, hist])

    if combine:
        DOS = np.mean(DOS, axis=0)

    
    if plot:
        # Filter DOS values where 0.05 < DOS[0] < 0.95
        if with_std:
            mask = (DOS[0] > 0.05) & (DOS[0] < 0.95)
            filtered_DOS = DOS[:, mask]
            energy_filtered = DOS[0][mask]
            dos_filtered = DOS[1][mask]
            
            # Calculate weighted mean of the energy
            mean_weighted = np.sum(energy_filtered * dos_filtered) / np.sum(dos_filtered)
            
            # Calculate weighted standard deviation
            variance_weighted = np.sum(dos_filtered * (energy_filtered - mean_weighted) ** 2) / np.sum(dos_filtered)
            std_weighted = np.sqrt(variance_weighted)
            # Add vertical lines for mean and mean +/- std
            plt.axvline(x=mean_weighted, color='r', linestyle='--', label='Mean')
            plt.axvline(x=mean_weighted + std_weighted, color='g', linestyle='--', label='Mean + Std')
            plt.axvline(x=mean_weighted - std_weighted, color='b', linestyle='--', label='Mean - Std')
            
        # Plotting
        plt.semilogy(DOS[0], DOS[1], '.')
        plt.xlabel('Energy')
        plt.ylabel('Density of States')
        
        
        plt.legend()
        plt.show()

    return DOS


def plot_3D_Entanglement(L, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, axis="x", disorder=[0, 0, 0], shift=False, filter=None, test=False):
    """
    Plot the entanglement spectrum for a 3D lattice.

    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    axis (str): Axis for entanglement cut.
    shift (bool): Shift the physical spectrum.
    filter (int): Filter the entanglement spectrum.
    test (bool): Test for correctness of the entanglement spectrum.

    Returns:
    None
    """

    if axis == "x":
        FT = "yz"
    elif axis == "y":
        FT = "xz"
    elif axis == "z":
        FT = "xy"

    assist = {"x": 0, "y": 1, "z": 2}

    H = Hamiltonian(L, PBC, t_z, t, t_0, m_z,
                    w=0, disorder=disorder)
    C = generate_C(H, L, axis=axis, FT=FT)
    vals_3D = C.subdiag(FT).real

    if test:
        C_check = generate_C(H, L, axis=axis, FT="y")
        test = np.linalg.eigvalsh(C.array)
        print("Correct eigenvalues1:", np.allclose(
            np.sort(vals_3D.flatten()), np.sort(np.linalg.eigvalsh(C_check.array)), atol=1e-10))
        print("Correct eigenvalues2:", np.allclose(
            np.sort(vals_3D.flatten()), np.sort(test), atol=1e-10))
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if filter is not None and filter < vals_3D.shape[2]:
        new_vals = np.zeros((vals_3D.shape[0], vals_3D.shape[1], filter))

        for i, j in product(range(vals_3D.shape[0]), range(vals_3D.shape[1])):
            new_vals[i, j] = vals_3D[i, j, (vals_3D.shape[2]-filter)//2:(vals_3D.shape[2]+filter)//2]
        vals_3D = new_vals

    if shift:
        vals_3D = np.concatenate(
            (vals_3D[:, vals_3D.shape[1]//2:],
             vals_3D[:, :vals_3D.shape[1]//2]), axis=1
        )
        vals_3D = np.concatenate(
            (vals_3D[vals_3D.shape[0]//2:, :],
             vals_3D[:vals_3D.shape[0]//2, :]), axis=0
        )

    for i in range(vals_3D.shape[0]):
        for j in range(vals_3D.shape[1]):
            for k in range(vals_3D.shape[2]):
                ax.scatter(j, i, vals_3D[i, j, k], color='b')

    ax.set_xlabel("k_" + FT[0])
    ax.set_ylabel("k_" + FT[1])
    ax.set_zlabel("Entanglement Spectrum")
    ax.set_xticks([0, L[assist[FT[0]]]//2, L[assist[FT[0]]]])
    ax.set_yticks([0, L[assist[FT[1]]]//2, L[assist[FT[1]]]])

    if shift:
        ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    else:
        ax.set_xticklabels(['0', '$\pi$', '$2\pi$'])
        ax.set_yticklabels(['0', '$\pi$', '$2\pi$'])

    plt.show()


def plot_with_shift(spectrum, ax, label, symbol="-", color="blue", shift=False, x_values=None):
    """
    Plot a spectrum with a shift.

    Parameters:
    spectrum (numpy.ndarray): Spectrum to be plotted.
    ax (matplotlib.axes._subplots.AxesSubplot): Axes for the plot.
    label (str): Label for the plot.
    symbol (str): Symbol for the plot.
    color (str): Color for the plot.
    shift (bool): Shift the spectrum.
    x_values (numpy.ndarray): x-values for the plot.

    Returns:
    None
    """

    if spectrum.ndim == 1:
        spectrum = spectrum.reshape(-1, 1)

    if x_values == None:
        if shift:
            x_values = np.linspace(-np.pi,
                                      np.pi, spectrum.shape[0], endpoint=False)
            spectrum = np.concatenate(
                (spectrum[spectrum.shape[0]//2:],
                 spectrum[:spectrum.shape[0]//2]), axis=0
            )

        else:
            x_values = np.linspace(0, 2*np.pi, spectrum.shape[0], endpoint=False)

        for i in range(spectrum.shape[1]):
            ax.plot(x_values, spectrum[:, i], symbol, label=label, color=color)
    
    else:
        print("not implemented yet")


def plot_localization(L, vecs, axis = "x", axes = None, energies = None):
    """
    Plot the localization of the eigenstates in 1D.
    
    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    vecs (numpy.ndarray): Eigenstates for the 3D lattice.
    axis (str): Axis for entanglement cut.
    axes (matplotlib.axes._subplots.AxesSubplot): Axes for the plot.
    energies (numpy.ndarray): Energies of the eigenstates.
    
    Returns:
    None
    """

    assert axis == "x"
    
    l = L[0]
    vals = np.zeros(l)
    
    if vecs.ndim == 2:

        # Compute the indices for accumulation
        indices = np.arange(vecs.shape[0]) % (2 * l) // 2

        # Use np.add.at for unbuffered in-place addition
        np.add.at(vals, indices, np.sum(np.abs(vecs)**2, axis=1))

    else:
        indices = np.arange(vecs.shape[1]) % (2 * l) // 2
        for i in range(vecs.shape[0]):
            np.add.at(vals, indices, np.sum(np.abs(vecs[i])**2, axis=1))

    if axes != None:
        if energies is not None:
            axes.plot(vals/np.sum(vals), 'o', label = f"E = {energies.real:.2f}")
        else:
            axes.plot(vals/np.sum(vals), 'o')
    else:
        plt.plot(vals/np.sum(vals), 'o')
        plt.show()


def get_entropy_scaling(N, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, FT="z", w=[0], N_max=1, axis="x", save=False):
    """
    Get the scaling of the entanglement entropy with the size of the lattice.

    Parameters:
    N (int): Maximum lattice size.
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    axis (str): Axis for entanglement cut.
    save (bool): Save the entropies.

    Returns:
    numpy.ndarray: Entanglement entropies for the lattice.
    """

    assert len(FT) == 1

    entropies = np.zeros(N//2)

    for i in range(N//2):
        L = [2*(i+1), 2*(i+1), 2*(i+1)]
        H = Hamiltonian(L, PBC=PBC, t_z=t_z, t=t, t_0=t_0,
                    m_z=m_z, w=w[0], disorder=[1, 0, 0])
        C = generate_C(H, L, axis=axis, FT=FT)
        for j in range(N_max-1):
            print(f"Iteration {j+2} of {N_max}")
            H = Hamiltonian(L, PBC=PBC, t_z=t_z, t=t, t_0=t_0,
                        m_z=m_z, w=w[0], disorder=[1, 0, 0])
            C = ((j+1)*C + generate_C(H, L, axis=axis, FT=FT))/(j+2)
        entvals = np.linalg.eigvalsh(C)
        print(C.ndim)
        entropies[i] = calc_entropy(entvals)

    if save:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = cwd + f"data\\{current_datetime}_entropies_N_max={N}_w={0}.h5"
        with h5py.File(filename, 'w') as f:
            f.create_dataset("Ns", data=np.arange(2, N+2, 2))
            f.create_dataset("entropies", data=entropies)
    
    return entropies


def plot_entropy_scaling(N, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, FT="z", w=[0], N_max=1, axis="x", save=False, after=False):
    """
    Plot the scaling of the entanglement entropy with the size of the lattice.
    
    Parameters:
    N (int): Maximum lattice size.
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    axis (str): Axis for entanglement cut.
    save (bool): Save the plot.
    after (bool): Use the precomputed entropies.
    
    Returns:
    None
    """

    if after:
        entropies = get_entropy_scaling(N, PBC=PBC, t_z=t_z, t=t, t_0=t_0, m_z=m_z, FT=FT, w=w, N_max=1, axis=axis, save=False)
        for i in range(1, N_max):
            entropies = (i*entropies + get_entropy_scaling(N, PBC=PBC, t_z=t_z, t=t, t_0=t_0, m_z=m_z, FT=FT, w=w, N_max=1, axis=axis, save=False))/(i+1)
        if save:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data\\{current_datetime}_entropies_N_max={N}_w={0}.h5"
            with h5py.File(filename, 'w') as f:
                f.create_dataset("Ns", data=np.arange(2, N+2, 2))
                f.create_dataset("entropies", data=entropies)
    else:
        entropies = get_entropy_scaling(N, PBC=PBC, t_z=t_z, t=t, t_0=t_0, m_z=m_z, FT=FT, w=w, N_max=N_max, axis=axis, save=save)

    x = 2*(np.arange(N//2)+1)

    # fit = np.polyfit(x, entropies, 2)

    plt.loglog(x, entropies/(x**2), 'o')
    # plt.plot(x, fit[0]*x**2 + fit[1]*x + fit[2])
    plt.xlabel("N")
    plt.ylabel("Entanglement Entropy/A")
    plt.show()


def plot_given_results(L, results, FT, shift, history, w_array, save=False, show=True, fast=False):
    """
    Plot the physical and entanglement spectra for a 3D lattice with disorder.
    
    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    results (list): List of results.
    FT (str): Fourier transform type.
    shift (bool): Shift the physical spectrum.
    history (bool): Plot the history of the entanglement spectrum.
    w_array (list): List of disorder strengths.
    save (bool): Save the plot.
    show (bool): Show the plot.
    fast (bool): Use the fast method for the entanglement spectrum.
    
    Returns:
    None
    """

    w_array = np.atleast_1d(w_array)
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 6))

    if history:
        for i, result in enumerate(results):
            if fast:
                w, eigvals, spectrum, DOS, entropies, eigvecs = result
            else:
                w, eigvals, spectrum, DOS, entropies = result
            axes[1, 1].plot(entropies, 'o',
                            label='Entanglement Entropies of all iterations')
            axes[2, 0].plot(DOS[0], DOS[1], '-',
                            label='Density of States of last iteration')
            axes[2, 1].plot(w, DOS[1][DOS[1].shape[0]//2], 'o',
                            label='Density of States at E = 0 of last iteration')
            
        if fast:
            physical_spectrum, entanglement_spectrum, entvecs = np.zeros((L[2], 2*L[0]*L[1])), np.zeros((L[2], L[0]*L[1])), np.zeros((L[2], L[0]*L[1], L[0]*L[1]), dtype=complex)
            for i in range(eigvals.shape[0]):
                ind = (L[0]*(i % L[1]))
                physical_spectrum[i//L[1], (2*ind):(2*(ind + L[0]))] = spectrum[i, :]
                entanglement_spectrum[i//L[1], ind:(ind + L[0])] = eigvals[i, :]
                entvecs[i//L[1], ind:(ind+L[0]), ind:(ind+L[0])] = eigvecs[i, :]
            
            
        plot_with_shift(physical_spectrum, axes[0, 0], shift=shift,
                        label='Physical Spectrum of last iteration', symbol='.')
        plot_with_shift(entanglement_spectrum, axes[0, 1], shift=shift,
                        label='Entanglement Spectrum of last iteration', symbol='.')
        if fast:
            rows, cols = np.where(np.abs(entanglement_spectrum - 0.5) < 499e-3) 
            indexpairs = list(zip(rows, cols))
            for i in range(10):
                random_index_pair = indexpairs[np.random.choice(len(indexpairs))]
                sorted_vecs = entvecs[np.ix_(np.atleast_1d(random_index_pair[0]), np.arange(entvecs.shape[1]), np.atleast_1d(random_index_pair[1]))]
                Ln = [L[0]//2, L[1], L[2]]
                plot_localization(Ln, sorted_vecs, axis="x", axes=axes[1, 0], energies = entanglement_spectrum[random_index_pair[0], random_index_pair[1]])
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('Localization of 1/2 eigenstates')
            axes[1, 0].legend()
        else:
            plot_with_shift(np.log((1-eigvals)/eigvals), axes[1, 0], shift=shift,   
                        label='Entanglement Energy of last iteration', symbol='-')

        axes[0, 0].set_xlabel(f'k_{FT}')
        axes[0, 0].set_ylabel('Physical Spectrum')
        axes[0, 0].set_ylim([-0.8, 0.8])

        axes[0, 1].set_xlabel(f'k_{FT}')
        axes[0, 1].set_ylabel('Entanglement Spectrum')

        axes[1, 1].set_xlabel('index')
        axes[1, 1].set_ylabel('Entanglement Entropy')

        axes[2, 0].set_xlabel('Energy')
        axes[2, 0].set_ylabel('Density of States')

        axes[2, 1].set_xlabel('index')
        axes[2, 1].set_ylabel('Density of States at E = 0')

        if shift:
            tick_positions = [-np.pi, 0, np.pi]
            tick_labels = ['$-\pi$', '0', '$\pi$']

        else:
            tick_positions = [0, np.pi, 2*np.pi]
            tick_labels = ['0', '$\pi$', '$2\pi$']

        for ax in axes.flat[:2]:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

        fig.suptitle('Disordered Spectra for w = ' + str(w_array))
    
    if save:
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"euler_plots\\{filename}.png")
        # plt.savefig(f"/cluster/home/vimohr/Master/Plots/{filename}.png")
        plt.close()

    if show:
        plt.show()


def export_data_to_h5py(L, PBC, t_z, t, t_0, m_z, FT, axis, N_max, num_bins, w_array):
    """
    Export the data to a HDF5 file.

    Parameters:
    L (tuple): Dimensions of the 3D lattice (Lx, Ly, Lz).
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    axis (str): Axis for entanglement cut.
    N_max (int): Number of iterations.
    num_bins (int): Number of bins for the density of states.
    w_array (list): List of disorder strengths.

    Returns:
    None
    """

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for w in w_array:
        filename = f"data\\{current_datetime}_results_N={L[0]}_w={w:.2f}.h5"
        with h5py.File(filename, 'w') as file:
                results = compute_eigenvalues(w, L, PBC, t_z, t, t_0, m_z, N_max, axis, FT, num_bins, [1, 0, 0], fast=True)
                file.create_dataset("w", data=w)
                file.create_dataset("avg_ES", data=results[1])
                file.create_dataset("last_PS", data=results[2])
                file.create_dataset("DOS", data=results[3])
                file.create_dataset("entropies", data=results[4])


def load_h5py_data(filename):
    """
    Load the data from a HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file.

    Returns:
    list: List of results.
    """
    with h5py.File(filename, "r") as file:
        print("Keys:", file.keys())
        if len(list(file.keys())) == 6:
            w = [np.array(file["w"])]
            eigvals = np.array(file["avg_ES"])
            spectrum = np.array(file["last_PS"])
            DOS = np.array(file["DOS"])
            entropies = np.array(file["entropies"])
            eigvecs = np.array(file["eigvecs"])
            return w, eigvals, spectrum, DOS, entropies, eigvecs
        
        
def band_closure(N, PBC, t_z, t, t_0, m_z, FT, axis, w, N_max, num_bins, disorder, fast=True):
    """
    Compute the band gap under changes in the lattice size.

    Parameters:
    N (int): Maximum lattice size.
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    axis (str): Axis for entanglement cut.
    w (float): Disorder strength.
    N_max (int): Number of iterations.
    num_bins (int): Number of bins for the density of states.
    disorder (list): Disorder directions.
    fast (bool): Use the fast method for the entanglement spectrum.

    Returns:
    None
    """

    if fast:
        gap = np.zeros(N//2)
        for n in 2*np.arange(1, N//2+1):
            print(f"Calculating gap for N = {n}")
            L = [n, n, n]
            H = Hamiltonian(L, PBC, t_z, t, t_0, m_z, w, disorder=disorder)
            C = generate_C(H, L, axis=axis, FT=FT, get_spectrum = False)
            entanglement_vals = np.linalg.eigvalsh(C)
            entanglement_spectrum = np.zeros((L[2], L[0]))
            for i in range(L[2]):
                entanglement_spectrum[i] = entanglement_vals[i*L[1]]
            diff = np.abs(entanglement_spectrum.flatten()-0.5)
            gap[n//2-1] = np.min(diff[np.where(diff > 1e-2)])
        x_values = np.linspace(0, 2*np.pi, entanglement_spectrum.shape[0], endpoint=False)
        for i in range(entanglement_spectrum.shape[1]):
            plt.plot(x_values, entanglement_spectrum[:, i], ".", color="blue")
        plt.show()
        plt.plot(2*np.arange(1, N//2+1), gap, "o")
        plt.show()
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data\\{current_datetime}_bandgap_N_max={N}_w={0}.h5"
        with h5py.File(filename, 'w') as f:
            f.create_dataset("Ns", data=np.arange(2, N+2, 2))
            f.create_dataset("gaps", data=gap)


def get_max_overlaps(V1, V2, triv=False):
    """
    Compute the maximum overlaps between two sets of vectors.
    
    Parameters:
    V1 (numpy.ndarray): First set of vectors.
    V2 (numpy.ndarray): Second set of vectors.
    triv (bool): Use the trivial method for the assignment problem.
    
    Returns:
    numpy.ndarray: Maximum overlaps between the two sets of vectors.
    """

    assert V1.shape == V2.shape
    starttime = time.time()
    overlaps = np.zeros((V1.shape[0], V1.shape[2], V2.shape[2]))
    if triv:
        for i in range(V1.shape[0]):
            overlaps[i] = np.eye(V1.shape[2])
        return overlaps
    for i in range(V1.shape[0]):
        matrix = np.abs(V1[i].T.conj() @ V2[i])**2
        cost = 1 - matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        overlaps[i][row_ind, col_ind] = 1
    print("Time taken:", time.time() - starttime)
    return overlaps


def average_eigvals(N, PBC, t_z, t, t_0, m_z, FT, axis, w, N_max, disorder, triv=False):
    L  = [N, N, N]
    H0 = Hamiltonian(L, PBC, t_z, t, t_0, m_z, w = 0, disorder=disorder)
    C0 = generate_C(H0, L, axis=axis, FT=FT)
    eigvalsh0, eigvecsh0 = np.linalg.eigh(C0)
    eigvallist = eigvalsh0
    for i in range(N_max):
        H = Hamiltonian(L, PBC, t_z, t, t_0, m_z, w, disorder=disorder)
        C = generate_C(H, L, axis=axis, FT=FT)
        eigvals, eigvecs = np.linalg.eigh(C)
        overlaps = get_max_overlaps(eigvecsh0, eigvecs, triv=triv)
        eigvallist = (i*eigvallist + np.einsum("nmp,np->nm", overlaps, eigvals))/(i+1)
    return eigvallist


def solve_assignment_problem(matrix):
    num_instances, num_classes = matrix.shape
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    x = {}
    for i in range(num_instances):
        for j in range(num_classes):
            x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # Constraints
    # Each instance is assigned to exactly one class
    for i in range(num_instances):
        solver.Add(sum(x[i, j] for j in range(num_classes)) == 1)
    
    # Each class is assigned to exactly one instance
    for j in range(num_classes):
        solver.Add(sum(x[i, j] for i in range(num_instances)) == 1)

    # Objective
    objective_terms = [matrix[j, i] * x[i, j] for i in range(num_instances) for j in range(num_classes)]
    solver.Maximize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()
    solution_matrix = np.zeros((num_classes, num_instances))
    
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(num_instances):
            for j in range(num_classes):
                if x[i, j].solution_value() == 1:
                    solution_matrix[j, i] = 1
        return solution_matrix
    
    else:
        print('The problem does not have an optimal solution.')


def plot_new_entanglement_spectrum(N, PBC="xyz", t_z=1, t=1, t_0=1, m_z=1, FT="z", w=[0], N_max=1, axis="x", save=False, triv=False):
    """
    Plot the entanglement spectrum for a 3D lattice with disorder.

    Parameters:
    N (int): Maximum lattice size.
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    FT (str): Fourier transform type.
    w (list): List of disorder strengths.
    N_max (int): Maximum number of iterations.
    axis (str): Axis for entanglement cut.
    save (bool): Save the plot.
    triv (bool): Use the trivial method for the assignment problem.

    Returns:
    None

    """
    
    L = [N, N, N]
    eigvallist = average_eigvals(N, PBC, t_z, t, t_0, m_z, FT, axis, w, N_max, [1, 0, 0], triv=triv)
    entanglement_spectrum = np.zeros((L[2], L[0]*L[1]))
    for i in range(eigvallist.shape[0]):
        ind = (L[0]*(i % L[1]))
        entanglement_spectrum[i//L[1], ind:(ind + L[0])] = eigvallist[i, :]
    if save:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data\\{current_datetime}_entanglement_spectrum_N={N}_w={w}_Nmax={N_max}.h5"
        with h5py.File(filename, 'w') as f:
            f.create_dataset("entanglement_spectrum", data=entanglement_spectrum)
    plot_with_shift(entanglement_spectrum, plt, shift=True, label='Entanglement Spectrum of last iteration', symbol='.')
    plt.show()


def combine_data():
    """
    Combine multiple h5py files into a single h5py file.
    """

    # Set the directory where your h5py files are stored
    folder_path = 'C:\\Users\\vinic\\Desktop\\Useful_Plots\\Physical_Spectra'

    # List all h5py files in alphanumeric order
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])

    # Initialize an empty list to store arrays
    all_data = []

    # Iterate over the files and load their content
    for file_name in file_names:
        with h5py.File(os.path.join(folder_path, file_name), 'r') as f:
            # Assuming each file contains one dataset named 'data'
            # Modify 'data' to the correct dataset name if necessary
            data = f['phyical_spectra'][:]
            print(data.shape)
            all_data.append(data)

    # Concatenate all arrays into a single array
    combined_data = np.array(all_data)

    # Save the combined array into a new h5py file
    output_file_path = 'data/Physical_spectra_all.h5'
    with h5py.File(output_file_path, 'w') as f:
        f.create_dataset('combined_data', data=combined_data)

    print(f"Combined data saved to {output_file_path}")

        
def plot_entropies():
    """
    Plot the entnanglement entropies and the entanglement spectrum for the 3D lattice with disorder.
    """
    with h5py.File("C:\\Users\\vinic\\Desktop\\Desktop_files\\Master Project\\data\\Entropies_all.h5", "r") as f:
        data = np.array(f["combined_data"])
        mean = np.mean(data[:, :, -1]/(60**2), axis=1)
        std = np.sqrt(np.var(data[:, :, -1]/(60**2), axis=1))
        x = np.arange(0, 8.2, 0.2)
        plt.plot(x, mean, '.', color="blue")
        plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.2)
        plt.plot(x, std, '.', color="red")
        plt.xlabel("w")
        plt.ylabel("Entanglement Entropy/A")
        plt.show()

    with h5py.File("C:\\Users\\vinic\\Desktop\\Desktop_files\\Master Project\\data\\Entanglement_spectra_all.h5" , "r") as f:
        data = np.array(f["combined_data"])
        y = np.zeros((data.shape[0], data.shape[1]))
        for i, a in enumerate(data):
            for j, b in enumerate(a):
                y[i, j] = calc_entropy(b, fast=True)
        mean = np.mean(y, axis=1)
        std = np.sqrt(np.var(y, axis=1))
        x = np.arange(0, 10.2, 0.2)
        plt.plot(x, mean/(60**2), '.', color="blue")
        plt.fill_between(x, (mean - std)/(60**2), (mean + std)/(60**2), color="blue", alpha=0.2)
        plt.plot(x, std/(60**2), '.', color="red")
        plt.xlabel("w")
        plt.ylabel("Entanglement Entropy/A")
        plt.show()


def calc_DOS(spectra, num_bins, plot=True, ky0 = False, ES=False, log=False, density=True):
    """
    Compute the density of states for the spectra.

    Parameters:
    spectra (numpy.ndarray): Spectra for the lattice.
    num_bins (int): Number of bins for the density of states.
    plot (bool): Plot the density of states.
    ky0 (bool): Use the ky = 0 slice.
    ES (bool): Use the entanglement spectrum.
    log (bool): Use a log scale for the plot.
    density (bool): Use the density of states.

    Returns:
    numpy.ndarray: Bins for the density of states.
    numpy.ndarray: Density of states.
    """
    shapes = spectra.shape
    print(shapes)
    if ky0:
        if ES:
            new_spectra = np.zeros((shapes[0], shapes[2], shapes[2]))
            for i in range(shapes[2]):
                new_spectra[:, i] = spectra[:, shapes[2]*i]
        else:
            new_spectra = np.zeros((shapes[0], shapes[2]//2, shapes[2]))
            for i in range(shapes[2]//2):
                new_spectra[:, i] = spectra[:, (shapes[2]//2)*i]
        spectra = new_spectra
    
    spectra = spectra.flatten()
    
    bins = np.linspace(np.min(spectra), np.max(spectra), num_bins)
    bins_center = (bins[1:] + bins[:-1])/2
    hist, bin_edges = np.histogram(spectra, bins=bins, density=density)
    if plot:
        if log:
            plt.semilogy(bins_center, hist/np.sum(hist), '.', color="blue")
        else:
            plt.plot(bins_center, hist/np.sum(hist), '.', color="blue")
        plt.show()
    return bins_center, hist


def get_spectrum(N, PBC, t_z, t, t_0, m_z, w, disorder, plot=False, fill=512):
    """
    Get the entanglement spectra for the lattice.

    Parameters:
    N (int): Maximum lattice size.
    PBC (str): Boundary conditions for the lattice.
    t_z (float): z-direction hopping parameter.
    t (float): xy-direction hopping parameter.
    t_0 (float): On-site potential.
    m_z (float): z-direction mass parameter.
    w (float): Disorder strength.
    disorder (list): Disorder directions.
    plot (bool): Plot the entanglement and physical spectra.
    fill (int): Number of fillings.

    Returns:
    numpy.ndarray: Entanglement spectrum.
    """

    H = Hamiltonian([N, N, N], PBC, t_z, t, t_0, m_z, w, disorder=disorder)
    C = generate_C(H, [N, N, N], axis="x", FT="z", fill=fill)
    eigvals = np.linalg.eigvalsh(C)
    if plot:
        new1, new2 = np.zeros((N, N**2)), np.zeros((N, N**2))
        for i in range(eigvals.shape[0]):
            ind1 = (N*(i % N))
            ind2 = (N*(i // N))
            new1[i//N, ind1:(ind1 + N)] = eigvals[i, :]
            new2[i%N, ind2:(ind2 + N)] = eigvals[i, :]
        plot_with_shift(new1, plt, shift=True, label=f'Entanglement Spectrum fill = {fill}', symbol='.')
        plt.show()
        plot_with_shift(new2, plt, shift=True, label=f'Physical Spectrum fill = {fill}', symbol='.')
        plt.show()    
    return eigvals


def plot_2D_entropy_scaling(data):
    """
    Plot the 2D heatmap for the entanglement entropy scaling.

    Parameters:
    data (numpy.ndarray): Entanglement entropies.

    Returns:
    None
    """

    mean = np.mean(data, axis=1)
    std = np.sqrt(np.var(data, axis=1))
    # Compute the matrix for the heatmap
    matrix = mean @ np.diag([1/x**2 for x in np.arange(2, 62, 2)])

    # Plot the main heatmap
    fig, ax = plt.subplots()
    sns.heatmap(matrix.T, cmap="YlGnBu", ax=ax)
    x_labels = np.arange(0, 10.2, 0.2)
    y_labels = np.arange(2, 62, 2)
    ax.set_xticks(np.linspace(0, matrix.shape[0]-1, len(x_labels[::5])))
    ax.set_xticklabels(x_labels[::5])
    ax.set_yticks(np.linspace(0, matrix.shape[1]-1, len(y_labels[::5])))
    ax.set_yticklabels(y_labels[::5])
    ax.set_xlabel("w")
    ax.set_ylabel("N")

    plt.show()
        



def main():
    """
    Main function for the script.
    """

    assist = {"x": 0, "y": 1, "z": 2}

    SAVE = True
    w_array = np.arange(32, 51)/5
    w_array = [0]
    disorder = [1, 0, 0]
    N = 40
    L = [N, N, N]
    PBC = "xyz"
    t_z = 3
    t = 1
    t_0 = 1
    m_z = 2
    FT = "z"
    axis = "x"
    shift = True
    num_bins = 40
    N_max = 100
    history = True

    get_spectrum(N, PBC, t_z, t, t_0, m_z, w_array[0], disorder, plot=True, fill=512)


if __name__ == "__main__":
    main()
