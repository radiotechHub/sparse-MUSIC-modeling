import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import find_peaks

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
    noise    = np.zeros((M, Nsnap, len_f), dtype=complex)
   
    
   
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

def MUSIC(R,K, A_scan):
    
    eigvals, eigvecs = np.linalg.eigh(R)
    Un = eigvecs[:, :-K]
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

def estimate_doa_fast(P, K, theta_scan):
    dtheta = np.abs(theta_scan[1] - theta_scan[0])
    min_distance_deg = 0.5
    min_distance_samples = int(np.round(min_distance_deg / dtheta))

    peaks, properties = find_peaks(
        P,
        distance=min_distance_samples
    )

    if len(peaks) == 0:
        return np.array([])

    peak_values = P[peaks]
    idx_sort = np.argsort(peak_values)[::-1]

    selected_peaks = peaks[idx_sort[:K]]
    selected_peaks = selected_peaks[np.argsort(theta_scan[selected_peaks])]

    theta_est = theta_scan[selected_peaks]

    return theta_est

def doa_rmse(theta_est, theta_true, penalty_deg=20):


    theta_est = np.asarray(theta_est)
    theta_true = np.asarray(theta_true)

    K = len(theta_true)

    if len(theta_est) < K:
        return penalty_deg

    theta_est = np.sort(theta_est[:K])
    theta_true = np.sort(theta_true)

    err = theta_est - theta_true

    return np.sqrt(np.mean(err**2))

def check_resolution(theta_est, theta_true, tol_deg=1.0):

    theta_est = np.asarray(theta_est)
    theta_true = np.asarray(theta_true)

    K = len(theta_true)

    if len(theta_est) < K:
        return 0

    theta_est = np.sort(theta_est[:K])
    theta_true = np.sort(theta_true)

    err = np.abs(theta_est - theta_true)

    if np.all(err <= tol_deg):
        return 1
    else:
        return 0

def main():
    """
    Monte-Carlo simulation of DOA estimation using MUSIC-based methods.
    Computes RMSE and Probability of Resolution vs SNR
    for ULA and sparse arrays.
    """
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
    SNR_db = np.arange(-20,20+2,2)
    
    
    
    theta_scan = np.arange(-30, 30.01, 0.01)
    array_sparse= np.array([0,1,6,9,11,13])
    array_index = np.arange(np.min(array_sparse),np.max(array_sparse)+1,1)
    
    
    Mc = 1000
    methods = ['MUSIC', 'FBSS', 'MF', 'MF+FBSS']
    N_methods = len(methods)
    
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
    
    
    A_scan_total  = steering_matrix(array_index,      theta_scan, f0, f0)
    A_scan_smooth = steering_matrix(array_index[0:L], theta_scan, f0, f0)
    
    
    RMSE = np.zeros((len(SNR_db), N_methods))
    PoR = np.zeros((len(SNR_db), N_methods))
    
    RMSE_sparse = np.zeros((len(SNR_db), N_methods))
    PoR_sparse = np.zeros((len(SNR_db), N_methods))
    # %% 
    for i_s in range (len(SNR_db)):
        snr_db = SNR_db[i_s]
        print("SNR_dB = ", snr_db)
        
        err_sum =       np.zeros(len(methods))
        resolut_sum =   np.zeros(len(methods))
        
        err_sum_sparse =       np.zeros(len(methods))
        resolut_sum_sparse =   np.zeros(len(methods))
        
        for i_mc in range (Mc):
            X_clean,X_noisy,X_sparse,t = generate_signal(
                array_index,angles_deg, array_sparse,f0,f_n,snr_db,Nt,signal_mode)
    
    
            idx = np.argmin(np.abs(f_n - f0))
            X_f0 = X_noisy[:, :, idx]
            
            
            R_single = X_f0 @ X_f0.conj().T / Nt
            R_fbss   = FBSS(R_single,L)
    
            R_f = np.zeros((len(array_index),len(array_index),len(f_n)), dtype=complex)
            for ik in range(len(f_n)):
                X_f = X_noisy[:,:,ik]
                R_f[:,:,ik] = X_f @ X_f.conj().T / Nt 
            R_mf     = np.sum(R_f,2)/len(f_n)
            R_mf_fbss = FBSS(R_mf,L)
            
            # %% MUSIC ULA
            P_single =      MUSIC(R_single,  K,A_scan_total)
            P_single_fbss = MUSIC(R_fbss,    K,A_scan_smooth)
            P_multi  =      MUSIC(R_mf,      K,A_scan_total)
            P_multi_fbss  = MUSIC(R_mf_fbss, K,A_scan_smooth)
            
            
            
            # %% Co-array reconstruct
    
            R_virt_single= sparse_reconstruct(X_sparse[:,:,idx], array_sparse,method)
            R_virt_fbss  = FBSS(R_virt_single,L)
    
            R_virt_mf = np.zeros((len(array_index),len(array_index),len(f_n)), dtype=complex)
            for ik in range(len(f_n)):
                R_virt_mf[:,:,ik] = sparse_reconstruct(X_sparse[:,:,ik], array_sparse,method)
            R_virt_mf_total     = np.mean(R_virt_mf, axis=2)
    
            R_virt_mf_fbss = FBSS(R_virt_mf_total,L)
    
            # %% sparse-MUSIC
            P_virt_single = MUSIC(R_virt_single,  K,A_scan_total)
            P_virt_fbss   = MUSIC(R_virt_fbss,    K,A_scan_smooth)
            P_virt_mf     = MUSIC(R_virt_mf_total,K,A_scan_total)
            P_virt_mf_fbss= MUSIC(R_virt_mf_fbss, K,A_scan_smooth)
            
            # %% Estimation
            theta_classic = estimate_doa_fast(P_single, K, theta_scan)
            theta_fbss = estimate_doa_fast(P_single_fbss, K, theta_scan)
            theta_mf = estimate_doa_fast(P_multi, K, theta_scan)
            theta_mf_fbss = estimate_doa_fast(P_multi_fbss, K, theta_scan)
            theta_all = [
                theta_classic,
                theta_fbss,
                theta_mf,
                theta_mf_fbss
                ]
            for i_m, theta_est in enumerate(theta_all):
                err_sum[i_m] += doa_rmse(theta_est, angles_deg, penalty_deg=20)**2
                resolut_sum[i_m] += check_resolution(theta_est, angles_deg, tol_deg=1.0)
            
            
            theta_classic_sparse = estimate_doa_fast(P_virt_single, K, theta_scan)
            theta_fbss_sparse    = estimate_doa_fast(P_virt_fbss, K, theta_scan)
            theta_mf_sparse      = estimate_doa_fast(P_virt_mf, K, theta_scan)
            theta_mf_fbss_sparse = estimate_doa_fast(P_virt_mf_fbss, K, theta_scan)
            theta_all_sparse = [
                theta_classic_sparse,
                theta_fbss_sparse,
                theta_mf_sparse,
                theta_mf_fbss_sparse
                ]
            for i_m, theta_est in enumerate(theta_all_sparse):
                err_sum_sparse[i_m] += doa_rmse(theta_est, angles_deg, penalty_deg=20)**2
                resolut_sum_sparse[i_m] += check_resolution(theta_est, angles_deg, tol_deg=1.0)
            
            
            
                
        RMSE[i_s, :] = np.sqrt(err_sum / Mc)
        PoR[i_s, :] = resolut_sum / Mc
        
        RMSE_sparse[i_s, :] = np.sqrt(err_sum_sparse / Mc)
        PoR_sparse[i_s, :] = resolut_sum_sparse / Mc
            
    # %% Graphs for ULA
    plt.figure("ULA: RMSE-SNR")
    for i_m, name in enumerate(methods):
        plt.semilogy(SNR_db, RMSE[:, i_m], marker='o', label=name)
    
    plt.grid(True)
    plt.xlabel('SNR, dB')
    plt.ylabel('RMSE, deg')
    plt.title('RMSE(SNR), log scale: ULA aperture')
    plt.legend()
    plt.show()  
    
    plt.figure("ULA: PoR-SNR")
    for i_m, name in enumerate(methods):
        plt.plot(SNR_db, PoR[:, i_m], marker='o', label=name)
    
    plt.grid(True)
    plt.xlabel('SNR, dB')
    plt.ylabel('Probability of Resolution')
    plt.title('PoR(SNR), ULA aperture')
    plt.legend()
    plt.show()      
    
    # %% Graphs for sparse
    plt.figure("Sparse: RMSE-SNR")
    for i_m, name in enumerate(methods):
        plt.semilogy(SNR_db, RMSE_sparse[:, i_m], marker='o', label=name)
    
    plt.grid(True)
    plt.xlabel('SNR, dB')
    plt.ylabel('RMSE, deg')
    plt.title('RMSE(SNR), log scale: Sparse aperture')
    plt.legend()
    plt.show()  
    
    plt.figure("Sparse PoR-SNR")
    for i_m, name in enumerate(methods):
        plt.plot(SNR_db, PoR_sparse[:, i_m], marker='o', label=name)
    
    plt.grid(True)
    plt.xlabel('SNR, dB')
    plt.ylabel('Probability of Resolution')
    plt.title('PoR(SNR), Sparse aperture')
    plt.legend()
    plt.show()      
           
    
# %% Code 
if __name__ == "__main__":
    main()