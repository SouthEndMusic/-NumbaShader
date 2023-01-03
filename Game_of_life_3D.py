# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:29:43 2022

author: Bart de Koning
"""

import numpy as np
from numba import cuda

from math import ceil

from NumbaShader3D import get_ray, get_background
from NumbaShader3D import Shader_viewer, Shader_cam
from Cube_grid     import cube_grid_intersect


@cuda.jit
def GOL_iter_kernel(grid_in,grid_out):
    """
    CUDA kernel for performing a Game of Life iteration.
    """
    
    i,j = cuda.grid(2)
    
    if (i >= grid_in.shape[0]) or (j >= grid_in.shape[1]):
        return
    
    for k in range(grid_in.shape[2]):
    
        neighbours = 0
        
        for i_ in (i-1, i+1):
            i_ = (i_ % grid_in.shape[0])
            
            for j_ in (j-1, j+1):
                j_ = (j_ % grid_in.shape[1])
                
                for k_ in (k-1, k+1):
                    k_ = (k_ % grid_in.shape[2])
                    
                    neighbours += grid_in[i_,j_,k_]
                                
        if grid_in[i,j,k]:
            
            if not (2 < neighbours < 6):
                next_val = False
            else:
                next_val = True
                
        else:
            
            if (2 < neighbours < 5):
                next_val = True
            else:
                next_val = False
    
        
        # Update grid
        grid_out[i,j,k] = next_val



class Game_of_life(object):
    
    def __init__(self,
                 grid_init):
        """
        A Wrapper around a 3D boolean array for performing
        game of life iterations on a 3D torus.
        """
        
        self.grid              = cuda.to_device(grid_init)
        self.threads_per_block = (16,16)
        self.blocks_per_grid   = (ceil(self.grid.shape[0]/self.threads_per_block[0]),
                                  ceil(self.grid.shape[1]/self.threads_per_block[1]))
        
        
    def iterate(self):
        """
        Perform a Game of Life iteration.
        """
        
        grid_out = cuda.device_array(self.grid.shape, dtype = bool)
        
        GOL_iter_kernel[self.blocks_per_grid,
                        self.threads_per_block](self.grid, grid_out)
        
        self.grid[:] = grid_out
        
        
        
        
@cuda.jit
def Game_of_life_shader_kernel(canvas, grid,
                               cam_loc, cam_rs, cam_up, cam_right, cam_back, rays):
   
    """Shader kernel for viewing Game of Life cube grid."""
    
    i,j = cuda.grid(2)

    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    ray = get_ray(i, j, rays[i,j], canvas.shape,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back)
    
    # Cube grid intersection test
    ax_intersect, intersect_found, grid_indices = cube_grid_intersect(grid, ray)
    
    if intersect_found:
        w = max((1-ray[6]/3,0))
        
        if ax_intersect == 0:
            color = (212/255,164/255,34/255) # blue
        elif ax_intersect == 1:
            color = (122/255,171/255,31/255) # green
        else:
            color = (227/255,230/255,76/255) # cyan
            
        canvas[i,j,0] = w*color[0]
        canvas[i,j,1] = w*color[1]
        canvas[i,j,2] = w*color[2]
            
    else:
        canvas[i,j] = get_background(ray, 0.5)
    
    
    
         
        
class GOL_viewer(Shader_viewer):
    
    def __init__(self,
                 grid,
                 **kwargs):
        """
        viewer of Game of Life grid.
        """          
        
        super().__init__(kernel = Game_of_life_shader_kernel,
                         **kwargs)
        
        self.GOL = Game_of_life(grid)
        
        self.action_keys['g'] = self.GOL.iterate
        
        self.winname = "Game of Life 3D viewer"
        
    def get_args(self):
        return [self.canvas, self.GOL.grid]
    
    
    
    def animation_1(self, N = 250, frames_per_iter = 10,
                    save_folder = None):
        
        import tqdm
        import cv2
        
        theta_vals = np.linspace(0,2*np.pi,N) + np.pi/4
        
        for i, theta in tqdm.tqdm(list(enumerate(theta_vals))):
            
            self.cam.theta = theta
            self.cam.update()
            
            if (i > 0) and ((i % frames_per_iter) == 0):
                self.GOL.iterate()
            
            self.update_canvas()
            
            cv2.imshow(self.winname, self.canvas_host)
            
            if save_folder:
                cv2.imwrite(f"{save_folder}/{i}.png", self.canvas_host)
                
            key = cv2.waitKey(self.min_frame_dur)
            
            if key == ord('q'):
                break
            
        cv2.destroyAllWindows()
            
            
    
    
    
def random_bool_grid(shape, p_true = 0.5):
    return np.random.choice([True,False],shape, p = (p_true,1-p_true))
        
        
        
        
if __name__ == "__main__":
    
    from Cube_grid import Menger_grid
    
    grid = Menger_grid(depth = 4)
    
    grid[1:-1,1:-1,1:-1] = False
    
    #grid = random_bool_grid((25,25,25), p_true = 0.15)
    cam  = Shader_cam(res = [500,500],
                      blur = True,
                      anaglyph_rc = False)
    
    
    
    GOL = GOL_viewer(grid, cam = cam)
    #GOL.main()
    GOL.animation_1(N = 1000, frames_per_iter = 10)   
        
        
        
        
        
        