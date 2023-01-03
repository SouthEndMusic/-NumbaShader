# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:24:34 2022

@author: bart1
"""

import numpy as np
from numba import cuda

from Shader_tools import get_ray, get_background, advance_ray, \
                            sphere_intersect, get_sphere_normal, \
                                    refract_ray, reflect_ray
from math import sqrt
                
from Shader_setup import Shader_viewer, Shader_cam
from Cube_grid    import Menger_grid, cube_grid_intersect


@cuda.jit
def get_color(ray, ax_intersect):
    
    norm = sqrt(ray[0]**2 +
                ray[1]**2 +
                ray[2]**2)
    
    b = max(0,1-norm/1.2)
    
    if ax_intersect == 0:
        color = (b,0.,0.)
    elif ax_intersect == 1:
        color = (0.,b,0.)
    elif ax_intersect == 2:
        color = (0.,0.,b)
    
    # R = b*(ray[2]+1)/2
    # G = b*(ray[0]+1)/2
    # B = b*(1 - (ray[0]+1)/2)
    
    # color = (B,G,R)
        
    return color



@cuda.jit
def Cube_grid_with_sphere_kernel(canvas, grid, sphere_center,
                                  cam_loc, cam_rs, cam_up, cam_right, cam_back):

    n_sphere = 2.4

    i,j = cuda.grid(2) 

    if (i < canvas.shape[0]) and (j < canvas.shape[1]):
        
        ray = get_ray(i,j,canvas.shape,
                      cam_loc, cam_rs, cam_up, cam_right, cam_back)
        
        # First cube grid intersection test
        ray, ax_intersect, intersect_grid = cube_grid_intersect(grid, ray)
        
        # Then sphere intersection test
        ray, intersect_sphere = sphere_intersect(ray, sphere_center, 0.2, True)
        


        if intersect_sphere:
        
            # Sphere normal at first sphere intersection           
            normal_1 = get_sphere_normal(ray, sphere_center)
            
            ###
            ##### Refraction
            ###
            
            # Refraction at first sphere intersection
            ray_refract = refract_ray(ray,normal_1,1.,n_sphere)
            
            # Second sphere intersection
            ray_refract, intersect_sphere = sphere_intersect(ray_refract, sphere_center, 0.2, False)
            
            # Sphere normal at second sphere intersection           
            normal_2 = get_sphere_normal(ray_refract, sphere_center)
            
            # Refraction at second sphere intersection
            ray_refract = refract_ray(ray_refract,normal_2,n_sphere,1.)
            
            # Try intersection with grid again
            ray_refract, ax_intersect, intersect_grid = cube_grid_intersect(grid, ray_refract)
            
            if intersect_grid:
                
                ray_refract      = advance_ray(ray_refract)
                color_refraction = get_color(ray_refract, ax_intersect)
                
            else:
                color_refraction = get_background(ray_refract, 0.5)
            
            ###
            ##### Reflection
            ###
            
            ray_reflect = reflect_ray(ray,normal_1)
            
            # Try intersection with grid again
            ray_reflect, ax_intersect, intersect_grid = cube_grid_intersect(grid, ray_reflect)
            
            if intersect_grid:
                
                ray_reflect      = advance_ray(ray_reflect)
                color_reflection = get_color(ray_reflect,ax_intersect)
                
            else:
                color_reflection = get_background(ray_reflect, 0.5)
                
            
            # Compute reflection proportion by Schlick's approximation
            R0    = ((1-n_sphere)/(1+n_sphere))**2
            inner = ray[3] * normal_1[0] + \
                    ray[4] * normal_1[1] + \
                    ray[5] * normal_1[2]
            f = R0 + (1-R0)*(1+inner)**5 
                
            canvas[i,j] = (f*color_reflection[0] + (1-f)*color_refraction[0],
                           f*color_reflection[1] + (1-f)*color_refraction[1],
                           f*color_reflection[2] + (1-f)*color_refraction[2])
            
            
        else:
                
            if intersect_grid:

                ray         = advance_ray(ray)
                canvas[i,j] = get_color(ray,ax_intersect) 
                
            else:
                canvas[i,j] = get_background(ray, 0.5)
                
                
                
                
                
class Cube_grid_with_sphere_viewer(Shader_viewer):
    
    def __init__(self, **kwargs):
        super().__init__(kernel = Cube_grid_with_sphere_kernel,
                          **kwargs)
        
        self.cam.dist = 4
        self.cam.update()
        self.winname = "Cube grid with sphere viewer"
        self.set_grid(np.random.choice([True,False], (5,5,5)))
        self.set_sphere_center(np.zeros(3, dtype = np.float64))
        
        self.action_keys['p'] = self.print_vec
        
    def get_args(self):
        return [self.canvas, self.grid, self.sphere_center]
    
    def set_grid(self, grid):
        self.grid = cuda.to_device(grid)
        
    def set_sphere_center(self, center):
        self.sphere_center = cuda.to_device(center)
        
    def print_vec(self):
        print(self.cam.vec_host)
        
        
    def animation_1(self, n_frames = 1000, savefolder = None):
        
        import cv2
        import tqdm
        
        self.cam.dist = 0.3
        self.cam.set_rs(0.3)
        
        for i,s in tqdm.tqdm(list(enumerate(np.linspace(0,1,n_frames)))):
            
            self.cam.theta = s*2*np.pi
            self.cam.update()
            
            sphere_center = np.zeros(3, dtype = np.float64)
            sphere_center[2] = 0.1*np.sin(s*4*np.pi)
            self.set_sphere_center(sphere_center)
            
            self.update_canvas()
            
            cv2.imshow(self.winname, self.canvas_host)

            key = cv2.waitKey(10)

            if key == ord('q'):
                break
            
            if savefolder:
                cv2.imwrite(f"{savefolder}/{i}.png", self.canvas_hos)
            
        cv2.destroyAllWindows()
        
        
        
        
        
if __name__ == "__main__":
    
    seed = np.random.choice([True,False],(3,3,3))
    seed[1,1,1] = False
    
    grid = Menger_grid(depth = 5, seed = seed)
    # size = grid.shape[0]
    
    # x = np.arange(size)
    # y = np.arange(size)
    # z = np.arange(size)

    # X,Y,Z = np.meshgrid(x,y,z)
    
    # R     = np.sqrt((X - size/2)**2 + (Y - size/2)**2 + (Z-size/2)**2)
    
    # grid &= (R < size/2)
    
    CGwsV = Cube_grid_with_sphere_viewer(cam = Shader_cam(res = [1000,1000],
                                                          anaglyph_rc = True))
    CGwsV.cam.eye_offset = 0.02
    
    CGwsV.set_grid(grid)
    # CGwsV.main()
    CGwsV.animation_1()#savefolder = r"Results\Sphere_in_sponge")  