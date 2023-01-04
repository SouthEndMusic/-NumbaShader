# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:51:01 2022

author: Bart de Koning
"""
import numpy as np
from numba import cuda

from math import ceil, isnan

import cv2

from tqdm import tqdm

@cuda.jit
def Laplace_kernel(U_in, U_out, fixed, BC_dirichlet):
    """
    If BC_Dirichlet = False, then the BC is Neumann (homogeneous)
    """
    
    i,j = cuda.grid(2)
    
    if not ((0 <= i < U_out.shape[0]) and (0 <= j < U_out.shape[1])):
        return
    
    if not isnan(fixed[i,j]):
        U_out[i,j] = fixed[i,j]
        return

        
    val_new = 0.
    
    i_ = i + 1
    
    if (i_ < U_out.shape[0]):
        val_new += U_in[i_,j]
    else:
        if not BC_dirichlet:
            U_out[i,j] = U_in[i-1,j]
            return
        
    i_ = i - 1
    
    if (i_ >= 0):
        val_new += U_in[i_,j]
    else:
        if not BC_dirichlet:
            U_out[i,j] = U_in[i+1,j]
            return
        
    j_ = j + 1
    
    if (j_ < U_out.shape[1]):
        val_new += U_in[i,j_]
    else:
        if not BC_dirichlet:
            U_out[i,j] = U_in[i,j-1]
            return
        
    j_ = j - 1
    
    if (j_ >= 0):
        val_new += U_in[i,j_]
    else:
        if not BC_dirichlet:
            U_out[i,j] = U_in[i,j+1]
            return
        
    val_new /= 4
    
    U_out[i,j] = val_new
    

def solve_Laplace(fixed, n_iters = 100, BC_Dirichlet = True,
                  threads_per_block = (16,16), init_value = 0.5):
    
    S = fixed.shape
    
    blocks_per_grid = (ceil(S[0]/threads_per_block[0]),
                       ceil(S[1]/threads_per_block[1]))
    
    U_in  = cuda.device_array(S, dtype = np.float32)
    U_out = cuda.device_array(S, dtype = np.float32)
    F     = cuda.to_device(fixed)
    
    U_in[:] = init_value
    
    for i in tqdm(range(n_iters)):
        
        U_out[:] = 0
    
        Laplace_kernel[blocks_per_grid,
                       threads_per_block](U_in, U_out, F, BC_Dirichlet)
        
        U_in[:] = U_out
        
    return U_in.copy_to_host()
        
    
    
    
    
    
    
    
    
        
if __name__ == "__main__":
    
    S     = (400,1000)
    fixed = np.zeros(S, dtype = np.uint8)
    fixed = cv2.putText(fixed, "F R A C T A L", (50,240), cv2.FONT_HERSHEY_TRIPLEX, 3.8, 255)
    fixed = fixed.astype(np.float32)
    fixed[fixed == 0] = np.nan
    fixed = cuda.to_device(fixed)
    
    out_host = solve_Laplace(fixed, n_iters = 25000, BC_Dirichlet = False)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(out_host)