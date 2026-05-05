import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
# %% Functions

def lag_weights(cnt_vals, mode):
    max_cnt = np.max(cnt_vals)

    methods = {
        'linear':  lambda: cnt_vals / max_cnt,
        'sqrt':    lambda: np.sqrt(cnt_vals / max_cnt),
        'inverse': lambda: max_cnt / np.maximum(cnt_vals, 1),
        'uniform': lambda: np.ones_like(cnt_vals)
    }

    if mode not in methods:
        raise ValueError('Unknown mode')

    return methods[mode]()


def steering_matrix(array_index, angles_deg, f, f0):
    c = 3e8

    lam0 = c / f0
    d0 = lam0 / 2
    pos_array = array_index * d0

    k_wave = 2 * np.pi * f / c

    A = np.exp(
        1j * k_wave * pos_array[:, None] *
        np.sin(np.deg2rad(angles_deg))[None, :]
    )

    return A

def generate_signal(array_index, angles_deg, array_sparse,f0, f_n, SNR_db, Nsnap,signal_mode):
    M = len(array_index)
    K = len(angles_deg)
    SNR_lin = 10**(SNR_db/10)
    len_f   = len(f_n)

    
    

    fs = 10 * np.max(f_n)
    t = np.arange(Nsnap) / fs
    
    if signal_mode == 'harmonic':
        phi_n = 2 * np.pi * np.random.rand(K, len_f)
        S_n = np.exp(
            1j * (2 * np.pi * f_n[None, None, :] * t[None, :, None]
                + phi_n[:, None, :]))
    elif signal_mode == 'random':
        S_n = (
            np.random.randn(K, Nsnap, len_f)
            + 1j*np.random.randn(K, Nsnap, len_f)
        ) / np.sqrt(2)
    else:
        raise ValueError("signal_mode should be 'harmonic' or 'random'")
        
        
        
    X_clean  = np.zeros((M, Nsnap, len_f), dtype=complex)
    X_noisy  = np.zeros((M, Nsnap, len_f), dtype=complex)
    

   
    
   
    for i_f in range(len_f):
        A = steering_matrix(array_index, angles_deg, f_n[i_f], f0)
        X_clean[:, :, i_f]   = A @ S_n[:, :, i_f]
        Signal_power = np.mean(abs(X_clean[:,:,i_f])**2)
        noise_var = Signal_power / SNR_lin
        noise = np.sqrt(noise_var/2) * (
    np.random.randn(M, Nsnap) + 1j*np.random.randn(M, Nsnap)
)
        X_noisy[:, :, i_f]    = X_clean[:, :, i_f] + noise
    X_sparse = X_noisy[array_sparse,:,:]
    
    return X_clean, X_noisy, X_sparse, t

def MUSIC(R,f,array,K,theta_scan):

    
    A_scan = np.exp(1j*np.pi*array[:,None] @ np.sin(np.deg2rad(theta_scan[None,:])))
    U,s,Vh = np.linalg.svd(R)
    Un     = U[:,K:]
    denom = np.sum(np.abs(Un.conj().T @ A_scan)**2, axis=0)

    P = 1/denom
    P_db = 10*np.log10(P/np.max(P))
    return P_db

def FBSS(Rx,L):  
    M = np.size(Rx,0)  
    R = (Rx+Rx.conj().T) / 2
    P = M - L + 1
    
    J = np.fliplr(np.eye(L))
    R_forward  = np.zeros((L,L), dtype=complex)
    R_backward = np.zeros((L,L), dtype=complex)
    
    for p in range(P):
        
        Rx_sub = R[p:p+L,p:p+L]
        R_forward = R_forward+Rx_sub
        R_backward= R_backward+ J @ Rx_sub.conj() @ J
    R_fbss = (R_forward+R_backward)/(2*P)
    return R_fbss
    
def sparse_reconstruct(X_sparse,array_sparse, method):
    Rxx = X_sparse @ X_sparse.conj().T / np.size(X_sparse,1)
    Rxx_valid = Rxx.ravel()
    
    diff_pos = array_sparse[:,None] - array_sparse[None,:]
    virtual_pos = np.arange(np.min(array_sparse),np.max(array_sparse)+1)
    num_virtual = len(virtual_pos)
    
    idx_offset = num_virtual - 1
    delta_idx = diff_pos.ravel() + idx_offset
    Nbins = 2 * num_virtual - 1
    valid = (delta_idx >= 0) & (delta_idx < Nbins)
    
    delta_idx_valid = delta_idx[valid].astype(int)
    Rxx_vals_valid = Rxx_valid[valid]
    
    
    
    sum_vals = np.zeros(Nbins, dtype=complex)
    cnt_vals = np.zeros(Nbins, dtype=float)
    
    np.add.at(sum_vals, delta_idx_valid, Rxx_vals_valid)
    np.add.at(cnt_vals, delta_idx_valid, 1)
    
    avg_vals = np.zeros(Nbins, dtype=complex)
    nz = cnt_vals > 0
    
    avg_vals[nz] = sum_vals[nz] / cnt_vals[nz]
    
    w = lag_weights(cnt_vals, method)
    avg_vals[nz] = avg_vals[nz] * w[nz]
    
    
    center = num_virtual - 1
    
    first_col = avg_vals[center:]
    first_row = avg_vals[center::-1]

    Rx_virt = toeplitz(first_col,first_row)
    return Rx_virt




def main():
    # %% Parameters
    f0 = 3e9
    Nt = 1000
    angles_deg = np.array([-5, 5])
    K = len(angles_deg)
    deltaF = 50e6
    Fn= 11
    L = 10
    k = np.arange(Fn) - (Fn - 1) / 2
    f_n = f0 + k * deltaF
    SNR_db = 10

    theta_scan = np.arange(-90, 90, 0.01)
    array_sparse= np.array([0,1,6,9,11,13]) #Warning. Use only non-hole array
    array_index = np.arange(np.min(array_sparse),np.max(array_sparse)+1,1)
    '''
    choosing a method for sparse reconstruction
    '''
    method = 'linear'
    # method = 'sqrt'
    # method = 'inverse'
    # method = 'uniform'
    '''
    signals are coherent or independance
    '''
    signal_mode = 'harmonic'
    # signal_mode = 'random'

    # %% Imput signal modeling


    X_clean,X_noisy,X_sparse,t = generate_signal(array_index,angles_deg, array_sparse,f0,f_n,SNR_db,Nt,signal_mode)


    idx = np.argmin(np.abs(f_n - f0))
    X_f0 = X_noisy[:, :, idx]
    # %% Covariance Matrix
    R_single = X_f0 @ X_f0.conj().T / Nt
    R_fbss   = FBSS(R_single,L)

    R_f = np.zeros((len(array_index),len(array_index),len(f_n)), dtype=complex)
    for ik in range(len(f_n)):
        X_1 = X_noisy[:,:,ik]
        R_f[:,:,ik] = X_1 @ X_1.conj().T / Nt 
    R_mf     = np.sum(R_f,2)/len(f_n)
    R_mf_fbss = FBSS(R_mf,L)

    # %% MUSIC Spectrum
    P_single =      MUSIC(R_single,f0,array_index,K,theta_scan)
    P_single_fbss = MUSIC(R_fbss,f0,array_index[0:L],K,theta_scan)
    P_multi  =      MUSIC(R_mf,f0,array_index,K,theta_scan)
    P_multi_fbss  = MUSIC(R_mf_fbss,f0,array_index[0:L],K,theta_scan)






    # %% Sparse Array 

    R_virt_single= sparse_reconstruct(X_sparse[:,:,idx], array_sparse,method)
    R_virt_fbss  = FBSS(R_virt_single,L)

    R_virt_mf = np.zeros((len(array_index),len(array_index),len(f_n)), dtype=complex)
    for ik in range(len(f_n)):
        R_virt_mf[:,:,ik] = sparse_reconstruct(X_sparse[:,:,ik], array_sparse,method)
    R_virt_mf_total     = np.mean(R_virt_mf, axis=2)

    R_virt_mf_fbss = FBSS(R_virt_mf_total,L)

    # %% Sparse MUSIC Spectrum
    P_virt_single = MUSIC(R_virt_single,f0,array_index,K,theta_scan)
    P_virt_fbss   = MUSIC(R_virt_fbss,f0,array_index[0:L],K,theta_scan)
    P_virt_mf     = MUSIC(R_virt_mf_total,f0,array_index,K,theta_scan)
    P_virt_mf_fbss= MUSIC(R_virt_mf_fbss,f0,array_index[0:L],K,theta_scan)


    # %% Graphs
    plt.figure('MUSIC ULA')
    plt.plot(theta_scan, P_single,label='MUSIC')
    plt.plot(theta_scan, P_single_fbss,label='FBSS')
    plt.plot(theta_scan, P_multi,label='Multi-Frequency (MF)')
    plt.plot(theta_scan, P_multi_fbss,label='MF + FBSS')
    for ik in angles_deg:
        plt.axvline(x=ik,color = 'black',linewidth = 1.5)
    plt.grid(True)
    plt.xlabel('Theta, angle')
    plt.ylabel('Power, dB')
    plt.title('MUSIC spectrum for ULA aperture')
    plt.legend()   
    plt.show()

    plt.figure('MUSIC Sparse')
    plt.plot(theta_scan, P_virt_single,label='MUSIC')
    plt.plot(theta_scan, P_virt_fbss,label='FBSS')
    plt.plot(theta_scan, P_virt_mf,label='Multi-Frequency (MF)')
    plt.plot(theta_scan, P_virt_mf_fbss,label='MF + FBSS')
    for ik in angles_deg:
        plt.axvline(x=ik,color = 'black',linewidth = 1.5)
    plt.grid(True)
    plt.xlabel('Theta, angle')
    plt.ylabel('Power, dB')
    plt.title('MUSIC spectrum for sparse aperture')
    plt.legend()  
    plt.show()

if __name__ == "__main__":
    main()



