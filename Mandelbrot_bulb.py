# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:07:06 2022

@author: bart1
"""

import numpy as np
from numba import cuda

from math import atan2, acos, sin, cos, sqrt, floor

from Shader_tools import get_ray, sphere_intersect, get_background, \
                           advance_ray, set_t_intersect
                           
from Shader_setup import Shader_viewer
                           
                           
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
    
    MV = Mandelbulb_viewer(cam = Shader_cam(res = [1000,1000]))
    MV.main()



# import tqdm

# # TODO: Ray reflections (within cube) for more Mandelbulb in view
# # TODO: Do not only show bulb itself, but 'mist' of other points where iteration takes a while 

                
#     def show_setup(self, decrease = 50, show_reflections = False):
        
#         import matplotlib.pyplot as plt
        
#         fig = plt.figure()
#         ax  = fig.add_subplot(projection = "3d")
        
#         def plot_dot(ax, loc, label, **kwargs):
#             ax.scatter([loc[0]],
#                        [loc[1]],
#                        [loc[2]],
#                        label = label,
#                        **kwargs)              
            
#         # Show center
#         center_cpu = self.cam.view_center
#         plot_dot(ax, center_cpu, 'center')
        
#         # Show cam
#         cam_loc_cpu = cp.asnumpy(self.cam.loc)
#         plot_dot(ax, cam_loc_cpu, 'cam_loc')
        
#         cam_back_cpu = cp.asnumpy(self.cam.back)
#         plot_dot(ax, cam_back_cpu, 'cam_back')
        
#         cam_up_cpu    = cp.asnumpy(self.cam.up)
#         cam_right_cpu = cp.asnumpy(self.cam.right)
        
#         r_x, r_y = self.cam.cam_rs
        
#         top_screen_mid = cam_loc_cpu + r_y*cam_up_cpu
#         plot_dot(ax, top_screen_mid, 'top_screen_mid')
        
#         right_screen_mid = cam_loc_cpu + r_x*cam_right_cpu
#         plot_dot(ax, right_screen_mid, 'right_screen_mid')
        
#         screen_corners = [cam_loc_cpu + r_x*cam_right_cpu + r_y*cam_up_cpu,
#                           cam_loc_cpu + r_x*cam_right_cpu - r_y*cam_up_cpu,
#                           cam_loc_cpu - r_x*cam_right_cpu - r_y*cam_up_cpu,
#                           cam_loc_cpu - r_x*cam_right_cpu + r_y*cam_up_cpu]
#         screen_corners = np.array(screen_corners + [screen_corners[0]])

#         ax.plot(screen_corners[:,0],
#                 screen_corners[:,1],
#                 screen_corners[:,2])
        
#         n_x, n_y = self.cam.res//decrease
        
#         for Dx in np.linspace(-r_x,r_x,n_x):
#             for Dy in np.linspace(-r_y,r_y,n_y):
                
                
#                 ray_orig = cam_loc_cpu + Dx * cam_right_cpu + Dy * cam_up_cpu
                
#                 points = np.stack([cam_back_cpu,
#                                    ray_orig])
                
#                 ax.plot(points[:,0],
#                         points[:,1],
#                         points[:,2],
#                         color = 'k', ls = ':')
                
#                 ray_dir  = ray_orig - cam_back_cpu
#                 ray_dir /= np.linalg.norm(ray_dir)
                
#                 cdiff    = center_cpu - ray_orig
#                 norm     = np.linalg.norm(cdiff)
#                 inner    = np.sum(cdiff * ray_dir)
#                 root_arg = inner**2 + self.render_R**2 - norm**2
                
#                 if root_arg < 0:
#                     continue
                
#                 root     = np.sqrt(inner**2 + self.render_R**2 - norm**2)
#                 t_min    = inner - root
#                 t_max    = inner + root
                
#                 for q in range(self.n_reflections+1 if show_reflections else 1):
                    
#                     render_min = ray_orig + ray_dir*t_min
#                     render_max = ray_orig + ray_dir*t_max
                    
#                     points = np.stack([render_min, render_max])
                    
#                     ax.plot(points[:,0],
#                             points[:,1],
#                             points[:,2],
#                             color = 'k')
                    
#                     # Reflected ray
#                     ray_orig += t_max * ray_dir
                    
#                     render_sphere_normal  = ray_orig - center_cpu
#                     render_sphere_normal /= np.linalg.norm(render_sphere_normal)
                    
#                     inner = np.inner(ray_dir, render_sphere_normal)
                    
#                     ray_dir -= 2*inner*render_sphere_normal
                    
#                     t_min = 0
#                     t_max = 2*inner
            

#         ax.legend()
#         ax.set_xlim(-2,2)
#         ax.set_ylim(-2,2)
#         ax.set_zlim(-2,2)
        
        
#     def animate(self, N = 250, save_folder = None, start = 0):
        
#         elements = list(enumerate(np.linspace(0,2*np.pi,N)))[start:]
        
#         for i,theta_offset in tqdm.tqdm(elements):
 
#             self.cam.theta = np.pi/4 + theta_offset
#             self.cam.update()
#             self.update_canvas()
#             canvas_gpu = np.flipud(cp.asnumpy(self.canvas))
#             cv2.imshow(self.winname, canvas_gpu)
            
#             if save_folder:
#                 cv2.imwrite(f"{save_folder}/{i}.png", canvas_gpu)
                
#             key = cv2.waitKey(1)
            
#             if key == ord('q'):
#                 break
            
#         cv2.destroyAllWindows()
        
            
        
        
        
# if __name__ == "__main__":
    
#     MV = Mandelbulb_viewer(cam = Shader_cam_host(res = [1000,1000]))
#     # MV.show_setup(decrease = 200)
    
    
#     # MV.main()
#     MV.animate(save_folder = "Results/Mandelbulb_frames", N = 500)
    
#     # MBV = Mandelbrot_bulb_viewer()
#     # MBV.n_iters_max = 10#250
#     # MBV.Dt          = 0.01#0.001
#     # # MBV.show_setup(decrease = 200, show_reflections = True)
    
#     # # MBV.animate(save_folder = "Results/Mandelbulb_frames", N = 500)
#     # # show_anim("Results/Mandelbulb_frames", 500, framedur = 40,
#     # #           write_name = "Animation")
    
#     # MBV.main()