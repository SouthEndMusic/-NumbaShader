# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:07:06 2022

@author: bart1
"""

import numpy as np
from numba import cuda

from math import atan2, acos, sin, cos, sqrt, floor

from NumbaShader3D import get_ray, sphere_intersect, get_background, \
                           advance_ray
                           
from NumbaShader3D import Shader_viewer
                           
                           
@cuda.jit
def Mandelbulb_iteration(x,y,z,r,x0,y0,z0,power):
    """
    Perform one iteration for the Mandelbulb with the formula given by
    White and Nylander: https://en.wikipedia.org/wiki/Mandelbulb
    
    """
    
    phi   = atan2(y,x)
    theta = acos(z/r)
    
    rn        = r**power
    thetan    = power*theta
    phin      = power*phi
    sinntheta = sin(thetan)
    cosntheta = cos(thetan)
    sinnphi   = sin(phin)
    cosnphi   = cos(phin)
    
    rn_sinntheta = rn*sinntheta
    
    x = rn_sinntheta*cosnphi + x0
    y = rn_sinntheta*sinnphi + y0
    z = rn*cosntheta         + z0
    
    r = sqrt(x*x + y*y + z*z)
    
    return x,y,z,r


@cuda.jit
def Mandelbulb_colour(r0,colours,cam_dist,render_R,t):
    """
    Get the colour of the Mandelbulb
    """
    
    c = r0/2 * (colours.shape[0]-1)
    s = c % 1
    k = int(floor(c))
    d = max(1-2*(t-(cam_dist-render_R))/(2*render_R),0)
    # d = 1
    
    colour_B = d*((1-s)*colours[k,0] + s*colours[k+1,0])
    colour_G = d*((1-s)*colours[k,1] + s*colours[k+1,1])
    colour_R = d*((1-s)*colours[k,2] + s*colours[k+1,2])

    return colour_B, colour_G, colour_R
                           

@cuda.jit
def Mandelbulb_kernel(canvas, power, render_R,
                      Dt, n_iters_max, colours, cam_dist,
                      cam_loc, cam_rs, cam_up, cam_right, cam_back, rays):
    """
    CUDA jit kernel for rendering the Mandelbrot bulb using ray marching
    """
    
    # Pixel indices
    i,j = cuda.grid(2)
    
    # Check whether pixel indices are within grid
    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    # Get ray for this pixel
    ray = get_ray(i, j, rays[i,j], canvas.shape,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back)
    
    # Compute intersections with the sphere centered at (0,0,0) with radius
    # render_R, within which ray-marching will be performed
    int_found = sphere_intersect(ray, (0.,0.,0.), render_R, False)
    t_max     = ray[6]
    ray[6]    = 1e10
    int_found = sphere_intersect(ray, (0.,0.,0.), render_R, True)
    t_min     = ray[6]

    
    if not int_found:
        canvas[i,j] = get_background(ray, 0.5)
        return

    t     = t_min
    advance_ray(ray)

    
    # Ray marching for the Mandelbulb
    while (t < t_max):
        
        x  = ray[0]
        y  = ray[1]
        z  = ray[2]
        r0 = sqrt(ray[0]**2 + ray[1]**2 + ray[2]**2)
        r  = r0
        n  = 0
        
        diverged = False
        
        while (n < n_iters_max) and (not diverged):
            
            x,y,z,r = Mandelbulb_iteration(x,y,z,r,ray[0],ray[1],ray[2],power)
        
            diverged = (r > 2)
            
            n += 1
            
            if n == n_iters_max:
            
                canvas[i,j] = Mandelbulb_colour(r0,colours,cam_dist,render_R,t)
                return
            
        t += Dt

        
        ray[6] = Dt
        advance_ray(ray)

    # No intersection with Mandelbulb found
    canvas[i,j] = get_background(ray, 0.5)  




class Mandelbulb_viewer(Shader_viewer):

    def __init__(self,
                 **kwargs):
        
        super().__init__(kernel = Mandelbulb_kernel,
                         **kwargs)
        
        self.winname     = "Mandelbulb viewer"
        self.n_iters_max = 10
        self.render_R    = 1.2
        self.power       = 8
        self.Dt_ray      = 0.01
        
        self.colours = np.stack([np.array([0,0,0]),
                                 np.array([0,0,0]),
                                 np.array([0.4,0,0]),
                                 np.array([1,0.8,0.8]),
                                 np.array([0,0,0]),
                                 np.array([0,0,0])]).astype(np.float16)
        
        self.colours = cuda.to_device(self.colours)
        
        
    def get_args(self):
        return self.canvas, self.power, self.render_R, \
                self.Dt_ray, self.n_iters_max, self.colours, self.cam.dist
        
    
    
if __name__ == "__main__":
    
    from Shader_setup import Shader_cam
    
    MV = Mandelbulb_viewer(cam = Shader_cam(res = [500,500]))
    MV.main()