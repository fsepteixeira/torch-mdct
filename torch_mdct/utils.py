import torch
import numpy as np
from scipy.signal import kaiser

def mdct_basis_(N):
    n0 = ((N//2) + 1) /2
    idx   = np.arange(0,N,1).reshape(N, 1)  
    kn    = np.multiply(idx + n0,(idx[:(N//2),:] + 0.5).T)
    basis = np.cos((2*np.pi/N)*kn)
    return torch.FloatTensor(basis.T)

def kbd_window_(win_len, filt_len, alpha=4):
    window = np.cumsum(kaiser(int(win_len/2)+1,np.pi*alpha))
    window = np.sqrt(window[:-1] / window[-1])

    if filt_len > win_len:
        pad =(filt_len - win_len) // 2
    else:
        pad = 0

    window = np.concatenate([window, window[::-1]])
    window = np.pad(window, (np.ceil(pad).astype(int), np.floor(pad).astype(int)), mode='constant')
    return torch.FloatTensor(window)[:,None]

