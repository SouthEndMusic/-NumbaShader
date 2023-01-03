# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:35:23 2022

@author: bart1
"""

import numpy as np
from numba import cuda

from Cube_grid import cube_grid_intersect
from Shader_setup import Shader_cam, Shader_viewer
from Shader_tools import get_ray, get_background

@cuda.jit
def Pencil_kernel(canvas,grid, pencil_types,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back, rays):
    
    i,j = cuda.grid(2)
    
    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    ray = get_ray(i,j, rays[i,j], canvas.shape,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back)
    
    # Cube grid intersection test
    ax_intersect, intersect_found, grid_indices = cube_grid_intersect(grid, ray)
    
    if intersect_found:
        b           = max((1-(ray[6]-2)/2,0))
        pencil_type = pencil_types[grid_indices]
        
        # for c in range(3):
        #     if c == pencil_type:
        #         canvas[i,j,c] = b
        #     else:
        #         canvas[i,j,c] = 0
        
        canvas[i,j] = b
        
        
    else:
        canvas[i,j] = get_background(ray, 0.5)  
        
        
        
class Pencil_viewer(Shader_viewer):
    
    def __init__(self, pencil_types,
                 **kwargs):
        
        super().__init__(kernel = Pencil_kernel,
                         **kwargs)
        
        self.pencil_types = cuda.to_device(pencil_types)
        self.set_grid(pencil_types < 10)
        
    def get_args(self):
        return [self.canvas, self.grid, self.pencil_types]
    
    def set_grid(self, grid):
        self.grid = cuda.to_device(grid)
        
        
        

if __name__ == "__main__":
    
    N            = 25
    step_size    = 5
    pencil_types = np.full((N,N,N), 10, dtype = np.uint8) # 10 means void
    
    pencil_types[:,1::step_size,::step_size] = 0
    pencil_types[::step_size,:,1::step_size] = 1
    pencil_types[1::step_size,::step_size,:] = 2
    
    PV = Pencil_viewer(pencil_types,
                       cam = Shader_cam(res = [1000,1000],
                                        dist = 4,
                                        anaglyph_rc = True))   
    PV.main()
    
    