import numpy as np
from numba import cuda

from math import ceil, copysign

from tqdm import tqdm

@cuda.jit
def three_body_fractal_kernel(Phi_all, Area_diffs_max, Dt, power):
    
    i,j = cuda.grid(2)
    
    if not ((0 <= i < Phi_all.shape[0]) and (0 <= j < Phi_all.shape[1])):
        return
    
    Forces   = cuda.local.array((3,2), np.float32)
    
    exponent = (power-1)/2
    
    # Calculate new velocities
    for a in range(3):          
        for b in range(3):
            if (b >= a):
                continue

            diff   = (Phi_all[i,j,b,0] - Phi_all[i,j,a,0], 
                      Phi_all[i,j,b,1] - Phi_all[i,j,a,1])
            normsq = diff[0]**2 + diff[1]**2 + 1e-6
            
            Forces[a,0] += diff[0]*normsq**exponent
            Forces[a,1] += diff[1]*normsq**exponent
            Forces[b,0] -= diff[0]*normsq**exponent
            Forces[b,1] -= diff[1]*normsq**exponent
            
    # Update locations
    for a in range(3):
        Phi_all[i,j,a,0] += Dt*Phi_all[i,j,3+a,0]
        Phi_all[i,j,a,1] += Dt*Phi_all[i,j,3+a,1]
        
    # Update velocities
    for a in range(3):
        Phi_all[i,j,3+a,0] += Dt*Forces[a,0]
        Phi_all[i,j,3+a,1] += Dt*Forces[a,1]
          
            
    # COM_x = (Phi[0,0] + Phi[1,0] + Phi[2,0])/3
    # COM_y = (Phi[0,1] + Phi[1,1] + Phi[2,1])/3

    diffpos1x = Phi_all[i,j,0,0] - Phi_all[i,j,1,0]
    diffpos2x = Phi_all[i,j,2,0] - Phi_all[i,j,1,0]
    diffpos1y = Phi_all[i,j,0,1] - Phi_all[i,j,1,1]
    diffpos2y = Phi_all[i,j,2,1] - Phi_all[i,j,1,1]
    
    diffvel1x = Phi_all[i,j,3,0] - Phi_all[i,j,4,0]
    diffvel2x = Phi_all[i,j,5,0] - Phi_all[i,j,4,0]
    diffvel1y = Phi_all[i,j,3,1] - Phi_all[i,j,4,1]
    diffvel2y = Phi_all[i,j,5,1] - Phi_all[i,j,4,1]
    
    B = diffpos1x*diffpos2y - diffpos2x*diffpos1y
    
    Area_diff = 0.5*copysign(1,B)*(diffvel1x*diffpos2y + diffpos1x*diffvel2y -
                                   (diffvel2x*diffpos1y + diffpos2x*diffvel1y))
    
    Area_diff = Area_diff**2
    
    
    #Area = 0.5*abs(diffpos1x*diffpos2y - diffpos2x*diffpos1y)
    
    Area_diffs_max[i,j] = max(Area_diff, Area_diffs_max[i,j])
    
    # diff1 = PHI[:,:,0] - PHI[:,:,2]
    # diff2 = PHI[:,:,1] - PHI[:,:,2]
    # inner = cp.sum(diff1*diff2, axis = 2)
    
    # Areas = cp.sum(cp.square(diff1), axis = 2)*cp.sum(cp.square(diff2), axis = 2) - inner**2
    
    
def get_trajectories_host(i,j, shape, Dt, n_timesteps):
    
    G = 1
    
    # Particle system state tensor
    Phi = np.zeros((n_timesteps+1,6,2), dtype = np.float32)
    
    r   = 2
    
    # Set initial conditions
    Phi[0,0,0] = -1
    Phi[0,1,0] =  1
    Phi[0,2,0] = r*(2*i/(shape[0]-1)-1)
    Phi[0,2,1] = r*(2*j/(shape[1]-1)-1)
    
    for k in range(1,n_timesteps+1):
        
        Phi[k] = Phi[k-1]
        
        # Calculate new velocities
        for a in range(3):          
            for b in range(3):
                if (a == b):
                    continue

                diff = Phi[k-1,b] - Phi[k-1,a]
                norm = np.linalg.norm(diff)
                Phi[k,3+a] += Dt*G*diff/(norm**2)

                
        # Update locations
        Phi[k,:3] += Dt*Phi[k-1,3:]
            
    return Phi
    
    
            
    
    
    


if __name__ == "__main__":
    
    import cupy as cp
    import cv2

    
    N        = 1000  # Number of pixels vertically
    M        = 2000  # Number of pixels horizontally
    n_bodies = 3
    dim      = 2
    power    = -0.5 # gravity proportional to r**power
    PHI      = cp.zeros((N,M,2*n_bodies,dim)) # Third axis contains first positions and
                                              # Then velocities
    
    r = 2
    
    center = (0,0)
    
    # Set initial conditions
    PHI[:,:,0,1] = -1
    PHI[:,:,1,1] =  1
    PHI[:,:,2,0] = cp.linspace(center[0]-r,   center[0]+r,N)[:,None]
    PHI[:,:,2,1] = cp.linspace(center[1]-M/N*r,center[1]+M/N*r,M)[None,:]
    
    threads_per_block = (16,16)
    blocks_per_grid   = (ceil(N/threads_per_block[0]),
                         ceil(M/threads_per_block[1]))
    
    timestep    = 0.001
    n_timesteps = 60000
    
    # fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    # out    = cv2.VideoWriter("three_body_prob4.mp4", fourcc, 60, (N,M))
    
    # Dists_min  = cp.full((N,M,n_bodies), 1e10)
    
    Area_diffs_max = cp.zeros((N,M))
    
    show_interval = 250
    
    k = 0
    
    while True:
        
        k += 1
        
        three_body_fractal_kernel[blocks_per_grid,
                                  threads_per_block](PHI, 
                                                     Area_diffs_max,
                                                     timestep,
                                                     power)
                                                     
        # v1  = PHI[:,:,0] - PHI[:,:,1] 
        # v1 /= cp.linalg.norm(v1, axis = 2, keepdims = True)
        
        # v2  = PHI[:,:,2] - PHI[:,:,1]
        # v2 /= cp.linalg.norm(v2, axis = 2, keepdims = True)
        
        # inner = cp.sum(v1*v2, axis = 2)
        # Angle = cp.arccos(inner)
        
        # if k == 0:
        #     Angle_first = Angle.copy()
        
        # Dists1 = np.linalg.norm(PHI[:,:,0] - PHI[:,:,2], axis = 2)
        # Dists2 = np.linalg.norm(PHI[:,:,1] - PHI[:,:,2], axis = 2)
        # Dists3 = np.linalg.norm(PHI[:,:,0] - PHI[:,:,1], axis = 2)

        
        # where_smaller1 = (Dists1 < Dists_min[:,:,0])
        # where_smaller2 = (Dists2 < Dists_min[:,:,1])
        # where_smaller3 = (Dists3 < Dists_min[:,:,2])
        
        # Dists_min[where_smaller1,0] = Dists1[where_smaller1]
        # Dists_min[where_smaller2,1] = Dists2[where_smaller2]
        # Dists_min[where_smaller3,2] = Dists3[where_smaller3] 
        
        if (k % show_interval == 0):
            
            print(k)
            
            Min = Area_diffs_max.min()
            Max = Area_diffs_max.max()
    
            im = cp.log(Area_diffs_max/Min)/cp.log(Max/Min)
            im = cp.asnumpy((255*(1-im)).astype(np.uint8))
    
            # im    = cp.asnumpy((255*(1-Areas_min/Areas_min.max())**10).astype(np.uint8))
            # im    = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            cv2.imshow('',im)
            
            # out.write(im)
            cv2.imwrite(f"Results/3body/{k//show_interval}.png",im)
            
        key = cv2.waitKey(10)
        
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    # out.release()
        
        
        
        # ax.imshow((Phi_host-Phi_host.min())**0.5, cmap = 'jet', extent = [-2,2,-2,2])