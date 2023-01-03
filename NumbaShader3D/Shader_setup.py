# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:02:39 2022

author: Bart de Koning

Basic implementation of shaders made with numba.cuda.
"""
import numpy as np
from numba import cuda

from math import ceil, sin
from time import perf_counter

import sys

from pathlib import Path

path_here  = str(Path(__file__).parent.resolve())

sys.path.append(path_here)

from Shader_tools import Defocus

class Shader_cam(object):
    
    def __init__(self,
                 dist:  float = 2.5,
                 depth: float = 1.5,
                 theta: float = np.pi/4,
                 phi:   float = np.pi/4,
                 res:   list  = [250,250],
                 r_y:   float = 0.5,
                 r_x:   float = None,
                 anaglyph_rc  : bool = False,
                 blur         : bool = False,
                 focus_dropoff: float = 7.5,
                 focus_depth  : float = 0.7):
        """Object for handeling the parameters of the shader camera.
        
        Inputs
        ------
            dist
                Distance from the camera to the view_center
            depth
                Distance from the screen center to the back of the camera
            theta,phi
                In combination with dist: polar coordinates for the location
                of the camera with respect to view_center
            res
                Resolution of the screen in pixels (x,y)
            r_x,r_y
                Size of the screen in world coordinates
            anaglyph_rc
                Bool that says whether the render should be in red-cyan 3D
            use_depth_field
                Bool that says wether an array must be constructed with t_intersect
                for each pixel
            focus_dropoff
                see Defocus in Shader_tools.py
        
        """
        
        self.dist = dist
        self.depth = depth
        self.theta = theta
        self.phi = phi
        self.res = np.array(res)
        self.set_rs(r_y, r_x = r_x)
        self.rays = cuda.device_array((*res,8), dtype = np.float32)

        self.view_center = np.zeros(3)
        
        self.anaglyph_rc = anaglyph_rc
        self.blur        = blur
        
        # In case of using defocuss, this number determines how fast the focus drops off away from the focus distance
        self.focus_dropoff = focus_dropoff
        self.focus_depth   = focus_depth
        
        # In case of anaglyph_rc, this attribute determines what distance
        # the cameras will be moved left and right
        self.eye_offset = 0.1

        self.update()
        
        
    def set_rs(self, r_y, r_x = None):
        
        # In case r_x is not provided, make sure the aspect ratio
        # of the screen agrees with the aspect ratio of the canvas
        if not r_x:
            r_x = r_y * self.res[0]/self.res[1]
            
        self.cam_rs_host = np.array([r_x, r_y], dtype=np.float64)
        self.cam_rs      = cuda.to_device(self.cam_rs_host)
 
    
    def update(self):
        """(Re)compute the camera data that gets passed to the kernel.
    
        Computed vectors
        ----------------
            vec
                The unit vector pointing from the view_center to the camera
            up
                The unit vector indicating the up direction of the screen in
                world coordinates
            right
                The unit vector indicating the right direction of the screen in
                world coordinates
            loc
                The location of the camera (screen center) in global world coordinates
            back
                The location of the back of the camera
        """
        
        sintheta = np.sin(self.theta)
        costheta = np.cos(self.theta)
        sinphi = np.sin(self.phi)
        cosphi = np.cos(self.phi)
        
        self.vec_host = np.array([sinphi*costheta,
                                  sinphi*sintheta,
                                  cosphi])
        
        b = 1/np.sqrt(1 - self.vec_host[2]**2)
        a = -self.vec_host[2]*b
        
        self.up_host = a*self.vec_host
        self.up_host[2] += b
        
        self.right_host = np.cross(self.up_host, self.vec_host)
        self.loc_host   = self.view_center + self.dist*self.vec_host
        self.back_host  = self.loc_host + self.depth*self.vec_host
        
        self.vec   = cuda.to_device(self.vec_host)
        self.up    = cuda.to_device(self.up_host)
        self.right = cuda.to_device(self.right_host)
        
        if self.anaglyph_rc:
            
            offset = self.eye_offset * self.cam_rs_host[1] * self.right_host
            
            self.loc_left_host  = self.loc_host - offset
            self.loc_right_host = self.loc_host + offset
            
            self.back_left_host  = self.back_host - offset
            self.back_right_host = self.back_host + offset
            
            
            
            self.loc_left  = cuda.to_device(self.loc_left_host)
            self.loc_right = cuda.to_device(self.loc_right_host)
            
            self.back_left  = cuda.to_device(self.back_left_host)
            self.back_right = cuda.to_device(self.back_right_host)
            
        else:
        
            self.loc  = cuda.to_device(self.loc_host)
            self.back = cuda.to_device(self.back_host)
            
            
    def get_vars(self):
        """Get the camera variables that get passed to the CUDA kernel."""
                            
        if self.anaglyph_rc:
            
            vars_left  = (self.loc_left, self.cam_rs, self.up, self.right, self.back_left,
                          self.rays)
            vars_right = (self.loc_right, self.cam_rs, self.up, self.right, self.back_right,
                          self.rays)
            
            return vars_left, vars_right
            
        else:
            return self.loc, self.cam_rs, self.up, self.right, self.back, self.rays
        
  
    
#####
######## Test kernel
#####
        
from Shader_tools import get_ray, sphere_intersect, get_background, \
                             advance_ray
        
@cuda.jit
def test_kernel(canvas, 
                cam_loc, cam_rs, cam_up, cam_right, cam_back, rays):
    """Test kernel for shading."""
    
    # Pixel indices
    i,j = cuda.grid(2)
    
    # Check whether pixel indices are within grid
    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    # Get ray for this pixel
    ray = get_ray(i, j, rays[i,j], canvas.shape,
                  cam_loc, cam_rs, cam_up, cam_right, cam_back)
    
    # Calculate intersection with unit sphere
    intersect_found = sphere_intersect(ray, (0.,0.,0.), 1., True)
    
    # If no intersection is found, give the pixel a background color
    if not intersect_found:
        canvas[i,j] = get_background(ray, 0.5)
        return
    
    t_intersect = ray[6]

    advance_ray(ray)

    # Create colour gradient on sphere
    brightness = (1-(t_intersect-1.5)/1.5)
    R = sin(5*(ray[0]+1))**2
    B = 1 - R
    G = sin(5*(ray[1]+1))**2

    canvas[i, j, 0] = B*brightness
    canvas[i, j, 1] = G*brightness
    canvas[i, j, 2] = R*brightness
    
    






        
class Shader_viewer(object):
    
    def __init__(self,
                 cam               = Shader_cam(),
                 kernel            = test_kernel,
                 threads_per_block = (16,16)):
        """
        Object for the usage in shaders, both interactive and for rendering
        animations.
        """
        
        self.cam    = cam
        self.kernel = kernel
        
        canvas_shape = (*cam.res[::-1], 3)
        
        if cam.anaglyph_rc:
            self.canvas_left  = cuda.device_array(canvas_shape, 
                                                  dtype = np.float16)
            self.canvas_right = cuda.device_array(canvas_shape, 
                                                  dtype = np.float16)
            
            self.canvas = None
        else:
            self.canvas = cuda.device_array(canvas_shape, dtype = np.float16)
        
        self.canvas_host = np.array(canvas_shape, dtype = np.uint8)
        
        self.threads_per_block = threads_per_block
        self.blocks_per_grid   = (ceil(cam.res[1]/self.threads_per_block[0]),
                                  ceil(cam.res[0]/self.threads_per_block[1]))
        
        self.winname = "Shader viewer"
        
        # Minimal frame duration in milliseconds
        self.min_frame_dur = 20
        
        # Frame duration
        self.Dt = 0
        
        
        # Main interaction loop action keys
        self.action_keys = dict(w=self.cam_up,
                                a=self.cam_left,
                                s=self.cam_down,
                                d=self.cam_right)
        
        
        
    def cam_up(self):
        """
        Move camera up (decrease phi)
        """
        self.cam.phi = max(1e-5, self.cam.phi - self.Dt)
        self.cam.update()

    def cam_down(self):
        """
        Move camera up (increase phi)
        """
        self.cam.phi = min(np.pi-1e-5, self.cam.phi + self.Dt)
        self.cam.update()

    def cam_left(self):
        """
        Move camer left (decrease theta)
        """
        self.cam.theta -= self.Dt
        self.cam.update()

    def cam_right(self):
        """
        Move camera right (increase theta)
        """
        self.cam.theta += self.Dt
        self.cam.update()



    def get_args(self):
        """
        Get the non-camera arguments (overload this method in a child class
                                      for passing more arguments)
        """
        args = [self.canvas]
        
        return args
    
    
    
    def update_canvas(self):
        """
        Run the shader kernel to fill the canvas and move to host
        """
        args = self.get_args()
        
        if self.cam.anaglyph_rc:
            cam_vars_left, cam_vars_right = self.cam.get_vars()
            
            # Left eye            self.canvas_left[:] = 0.
            args[0] = self.canvas_left
            
            self.kernel[self.blocks_per_grid,
                        self.threads_per_block](*args,
                                                *cam_vars_left)                  
                                                
            if self.cam.blur:
                self.canvas_left[:] = Defocus(self.canvas_left,
                                              self.cam.rays[:,:,6],
                                              self.cam.focus_depth,
                                              self.cam.focus_dropoff,
                                              self.blocks_per_grid,
                                              self.threads_per_block)

                                                
            # Right eye
            args[0] = self.canvas_right
            
            self.kernel[self.blocks_per_grid,
                        self.threads_per_block](*args,
                                                *cam_vars_right)
                                                
            
                                                
            if self.cam.blur:
                self.canvas_right[:] = Defocus(self.canvas_right,
                                               self.cam.rays[:,:,6],
                                               self.cam.focus_depth,
                                               self.cam.focus_dropoff,
                                               self.blocks_per_grid,
                                               self.threads_per_block)
                                                
                                                
                                                
            canvas_left_host  = self.canvas_left.copy_to_host()
            canvas_right_host = self.canvas_right.copy_to_host()
            
            self.canvas_host = np.zeros((*self.cam.res[::-1], 3), dtype=np.float16)
                                                
            self.canvas_host[:,:,2]  = canvas_left_host[:,:,2]
            self.canvas_host[:,:,:2] = canvas_right_host[:,:,:2]
            
            self.canvas_host = np.flipud((255*self.canvas_host).astype(np.uint8))
            
        else:
            self.kernel[self.blocks_per_grid,
                        self.threads_per_block](*args,
                                                *self.cam.get_vars())
                                                
            if self.cam.blur:
                
                self.canvas[:] = Defocus(self.canvas,
                                         self.cam.rays[:,:,6],
                                         self.cam.focus_depth,
                                         self.cam.focus_dropoff,
                                         self.blocks_per_grid,
                                         self.threads_per_block)
        
            self.canvas_host = np.flipud((255*self.canvas.copy_to_host()).astype(np.uint8))
        
        
        
    def main(self):
        """
        Interactive loop evaluating kernel.
        """
        
        import cv2
        
        while True:
            t_start = perf_counter()

            self.update_canvas()
                
            key = cv2.waitKey(self.min_frame_dur)
            
            cv2.imshow(self.winname, self.canvas_host)

            if key == ord('q'):
                break

            t_end   = perf_counter()
            self.Dt = t_end - t_start

            for action_key, action in self.action_keys.items():
                if key == ord(action_key):
                    action()

        cv2.destroyAllWindows()
        
        
        
    
if __name__ == "__main__":
    
    # Interactive test: use w,a,s,d to move the camera around
    SV = Shader_viewer(cam = Shader_cam(res = [500,500]))
    SV.main()