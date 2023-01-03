# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:05:05 2022

author: Bart de Koning
"""

import numpy as np
from numba import cuda

from NumbaShader3D import get_ray, get_background, argmin, advance_ray, \
                            translate_ray, cube_intersect
                            
from NumbaShader3D import Shader_viewer, Shader_cam
                            
from math import floor       


@cuda.jit
def get_Dts(ray, cube_size, prev_ax):
    """
    Get the times needed to intersect with the next cube in the cube grid
    in all 3 axes.
    """

    if prev_ax == 0:
        Dt_x = cube_size/abs(ray[3])
        
    else: 
        if ray[3] < 0:
            Dt_x = -(ray[0] % cube_size)/ray[3]
        elif ray[3] > 0:
            Dt_x = (cube_size - (ray[0] % cube_size))/ray[3]
        else:
            Dt_x = 1e10
            
        if Dt_x == 0:
            Dt_x = cube_size/ray[3]
            
            
    if prev_ax == 1:
        Dt_y = cube_size/abs(ray[4])
        
    else: 
        if ray[4] < 0:
            Dt_y = -(ray[1] % cube_size)/ray[4]
        elif ray[4] > 0:
            Dt_y = (cube_size - (ray[1] % cube_size))/ray[4]
        else:
            Dt_y = 1e10
            
        if Dt_y == 0:
            Dt_y = cube_size/ray[4]
            
            
    if prev_ax == 2:
        Dt_z = cube_size/abs(ray[5])
        
    else: 
        if ray[5] < 0:
            Dt_z = -(ray[2] % cube_size)/ray[5]
        elif ray[5] > 0:
            Dt_z = (cube_size - (ray[2] % cube_size))/ray[5]
        else:
            Dt_z = 1e10
            
        if Dt_z == 0:
            Dt_z = cube_size/ray[5]
            
    return Dt_x, Dt_y, Dt_z
    


@cuda.jit
def copy_ray(ray):
    return (ray[0],ray[1],ray[2],ray[3],ray[4],ray[5],ray[6],ray[7])


@cuda.jit
def cube_grid_intersect(grid, ray):
    
    grid_size = grid.shape[0]
    cube_size = 2/grid_size 
    
    # Compute intersection with bounding box cube
    ray_old               = copy_ray(ray)
    ax_intersect, in_cube = cube_intersect(ray)
    
    # No intersection with bounding box cube
    if (ax_intersect == -1) and (not in_cube):
        intersect_found = False
        grid_indices    = (0,0,0)            
        return ax_intersect, intersect_found, grid_indices

    
    
    # Translate ray so that (0,0,0) is a corner of the bounding box cube
    translate_ray(ray, (1.,1.,1.))
    
    
    if in_cube:
        # Advance ray to next intersection
        Dts          = get_Dts(ray, cube_size, ax_intersect)
        ax_intersect = argmin(Dts)
        ray[6]       = Dts[ax_intersect]


    # Advance ray to intersection point
    t_int = ray[6]
    advance_ray(ray)
    
    
    # Get indices within the grid
    idx_x = min(int(floor(ray[0]/cube_size)),grid_size-1)
    idx_y = min(int(floor(ray[1]/cube_size)),grid_size-1)
    idx_z = min(int(floor(ray[2]/cube_size)),grid_size-1)
    
    # Get ray direction signs in all dims
    ray_vel_x_pos = (ray[3] > 0)
    ray_vel_y_pos = (ray[4] > 0)
    ray_vel_z_pos = (ray[5] > 0)
    
    if in_cube:
        if ax_intersect == 0:
            idx_x = int(round(ray[0]/cube_size))
            
            if not ray_vel_x_pos:
                idx_x -= 1
            
        elif ax_intersect == 1:
            idx_y = int(round(ray[1]/cube_size))
            
            if not ray_vel_y_pos:
                idx_y -= 1
            
        elif ax_intersect == 2:
            idx_z = int(round(ray[2]/cube_size))
            
            if not ray_vel_z_pos:
                idx_z -= 1
    
    if not in_cube:
        if ax_intersect == 0:           
            idx_x = 0 if ray_vel_x_pos else grid_size-1
            
        elif ax_intersect == 1:            
            idx_y = 0 if ray_vel_y_pos else grid_size-1
            
        elif ax_intersect == 2:            
            idx_z = 0 if ray_vel_z_pos else grid_size-1

    
    
    if grid[idx_x,idx_y,idx_z]:
        intersect_found = True
        ray[:] = ray_old
        ray[6] = t_int
        grid_indices = (idx_x,idx_y,idx_z)
        return ax_intersect, intersect_found, grid_indices
    
  
    # Find the first cube intersection 
    while True:

        Dts          = get_Dts(ray, cube_size, ax_intersect)
        ax_intersect = argmin(Dts)
        Dt           = Dts[ax_intersect]
        ray[6]       = Dt
        advance_ray(ray)
        
        t_int += Dt
        
        
        if t_int > ray_old[6]:
            intersect_found = False
            grid_indices = (0,0,0)
            return ax_intersect, intersect_found, grid_indices
        
        
        
        if ax_intersect == 0:
            
            if ray_vel_x_pos:
                idx_x += 1
                
                # Check whether the ray has left the bounding box
                if idx_x == grid_size:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            else:
                idx_x -= 1
                
                # Check whether the ray has left the bounding box
                if idx_x == -1:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            if grid[idx_x,idx_y,idx_z]:
                intersect_found = True
                ray[:] = ray_old
                ray[6] = t_int
                grid_indices = (idx_x,idx_y,idx_z)
                return ax_intersect, intersect_found, grid_indices
            
            
        elif ax_intersect == 1:
            
            if ray_vel_y_pos:
                idx_y += 1
                
                # Check whether the ray has left the bounding box
                if idx_y == grid_size:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            else:
                idx_y -= 1
                
                # Check whether the ray has left the bounding box
                if idx_y == -1:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            if grid[idx_x,idx_y,idx_z]:
                intersect_found = True
                ray[:] = ray_old
                ray[6] = t_int
                grid_indices = (idx_x,idx_y,idx_z)                  
                return ax_intersect, intersect_found, grid_indices
            
        elif ax_intersect == 2: # Unnessecary check?
            
            if ray_vel_z_pos:
                idx_z += 1
                
                # Check whether the ray has left the bounding box
                if idx_z == grid_size:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            else:
                idx_z -= 1
                
                # Check whether the ray has left the bounding box
                if idx_z == -1:
                    intersect_found = False
                    grid_indices = (0,0,0)
                    return ax_intersect, intersect_found, grid_indices
                
            if grid[idx_x,idx_y,idx_z]:
                intersect_found = True
                ray[:] = ray_old
                ray[6] = t_int
                grid_indices = (idx_x,idx_y,idx_z)                 
                return ax_intersect, intersect_found, grid_indices






@cuda.jit
def Cube_grid_kernel(canvas, grid,
                     cam_loc, cam_rs, cam_up, cam_right, cam_back, rays):
    """Shader kernel for viewing a cube grid."""
    
    i,j = cuda.grid(2)

    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    ray = get_ray(i,j, rays[i,j], canvas.shape,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back)
    
    # Cube grid intersection test
    ax_intersect, intersect_found, grid_indices = cube_grid_intersect(grid, ray)
    
    if intersect_found:
        b = max((1-ray[6]/4,0))
        # canvas[i,j,ax_intersect] = b
        canvas[i,j] = b
    else:
        canvas[i,j] = get_background(ray, 0.5)
        
    

def Menger_grid(depth = 3, seed = None):
    """
    Create a boolean grid of the Menger sponge
    or a variation on this with a custom seed.
    """
    
    if seed is None:
        
        seed = np.ones((3,3,3), dtype = bool)
        
        seed[1,1,:] = False
        seed[1,:,1] = False
        seed[:,1,1] = False
    
    
    grid = np.ones(seed.ndim*[1], dtype = bool)
    
    for i in range(depth):
        grid = np.kron(grid,seed)
        
    return grid        
        



class Cube_grid_viewer(Shader_viewer):
    
    def __init__(self, kernel = Cube_grid_kernel,
                 **kwargs):
        
        super().__init__(kernel = kernel,
                         **kwargs)
        
        self.cam.dist = 4
        self.cam.update()
        self.winname = "Cube grid viewer"
        self.set_grid(np.random.choice([True,False], (5,5,5)))
        
    def get_args(self):
        return [self.canvas, self.grid]
    
    def set_grid(self, grid):
        self.grid = cuda.to_device(grid)
        
    def animation_1(self, N = 250, savefolder = None):
        
        import tqdm
        import cv2
        
        for i,s in tqdm.tqdm(list(enumerate(np.linspace(0,1,250)))):
            
            self.cam.theta = s*np.pi*2
            self.cam.update()
            
            self.update_canvas()           
            cv2.imshow(self.winname, self.canvas_host)

            key = cv2.waitKey(10)

            if key == ord('q'):
                break
            
            if savefolder:
                cv2.imwrite(f"{savefolder}/{i}.png", self.canvas_host)
            
        cv2.destroyAllWindows()
        
        
    def animation_2(self, savefolder = None):
        
        import tqdm
        import cv2
        
        seed = np.ones(5*[3], dtype = bool)
        seed[1,1,1,1,:] = False
        seed[1,1,1,:,1] = False
        seed[1,1,:,1,1] = False
        seed[1,:,1,1,1] = False
        seed[:,1,1,1,1] = False
        
        grid = Menger_grid(depth = 3, seed = seed)
        
        for i in tqdm.tqdm(range(grid.shape[0])):
            
            self.set_grid(grid[i,i])
            
            self.update_canvas()           
            cv2.imshow(self.winname, self.canvas_host)

            key = cv2.waitKey(0)

            if key == ord('q'):
                break
            
            if savefolder:
                cv2.imwrite(f"{savefolder}/{i}.png", self.canvas_host)
            
        cv2.destroyAllWindows()
            
        
        
        
        
def grid_from_text(text, sidelength = 250,
                   fontscale = 1,
                   org = (0,0),
                   font = None,
                   thickness = 10):
    
    import cv2
    
    if not font:
        font = cv2.FONT_HERSHEY_TRIPLEX
    
    canvas = np.zeros(2*[sidelength], dtype = np.uint8)
    
    canvas = cv2.putText(canvas, text, org, font, fontscale, 255)
    
    
    grid = np.zeros(3*[sidelength], dtype = bool)
    grid[:,:,
         (sidelength-thickness)//2:(sidelength+thickness)//2] = (canvas > 0)[:,:,None]
    
    return grid
    
    
    
    
        
def Menger_sponge_3D_test():
    
    CGV = Cube_grid_viewer(cam = Shader_cam(res         = [500, 500],
                                            anaglyph_rc = True,
                                            blur        = False,
                                            focus_depth = 2))
    
    CGV.cam.eye_offset = 0.2
    
    grid = Menger_grid(depth = 4, seed = None)
    CGV.set_grid(grid)
    
    CGV.main()
    
    
    
    
def text_test():
    thickness = 100
    
    # Make a Boolean grid depicting some text
    grid1 = grid_from_text("Non-Euclidean Dreams", sidelength = 500,
                          org = (8,120), fontscale = 1.2, thickness = thickness)
    grid2 = grid_from_text("&", sidelength = 500, 
                          org = (240,240), fontscale = 1.2, thickness = thickness)
    grid3 = grid_from_text("Fractal Realities", sidelength = 500,
                          org = (70,360), fontscale = 1.2, thickness = thickness)

    grid = grid1 | grid2 | grid3
    
    # import matplotlib.pyplot as plt
    # plt.imshow(grid[:,:,250])
    
    CGV = Cube_grid_viewer(cam = Shader_cam(res         = [500, 500],
                                            anaglyph_rc = True,
                                            blur        = False,
                                            focus_depth = 2))
    
    cam = CGV.cam
    cam.eye_offset = 0.05
    cam.dist = 2
    cam.phi = 0.001
    cam.theta = 0.001
    cam.update()
    
    
    CGV.set_grid(grid)
    
    CGV.main()
    
    
    
        
        
if __name__ == "__main__":
    
    # CGV = Cube_grid_viewer(cam = Shader_cam(res         = [500, 500],
    #                                         anaglyph_rc = False))
    # CGV.animation_2()
    
    Menger_sponge_3D_test() # Move camera with w,a,s,d
    #text_test()
