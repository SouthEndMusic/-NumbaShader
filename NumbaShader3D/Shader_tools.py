# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:04:25 2022

author: Bart de Koning

Tools for computations with rays.
"""

import numpy as np
from numba import cuda
from math import sqrt, sin, ceil

@cuda.jit
def normalize(vec):
    """
    Normalize a vector
    """
        
    norm = sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        
    vec_new = (vec[0]/norm,
               vec[1]/norm,
               vec[2]/norm)
    
    return vec_new
        

@cuda.jit
def get_ray(i, j, ray, shape,
            cam_loc, cam_rs, cam_up, cam_right, cam_back):
    """
    Compute the origin and the direction of the ray given by the input
    camera parameters.
    
    Inputs
    ------
        i,j
            The indices of the pixel corresponding to this ray on the canvas
        shape
            The shape of the canvas (y,x,3)
        cam_loc
            The absolute location of the camera
        cam_rs
            The size of the screen in world coordinates (r_x,r_y)
        cam_up
            The unit vector in world coordinates denoting the up direction of the camera
        cam_right
            The unit vector in world coordinates denoting the right direction fo the camera
        cam_back
            The absolute location of the back of the camera, determines the perspective;
            the direction of a ray is given by the line-segment from cam_back to the
            location at which the ray leaves the screen
    """
    
    # The offset in world coordinates from the middle of the screen of the pixel given by i,j
    Dx = (2*j/(shape[1]-1) - 1)*cam_rs[0]
    Dy = (2*i/(shape[0]-1) - 1)*cam_rs[1]
    
    # The location at which the ray leaves the escreen
    ray[0] = cam_loc[0] + Dy*cam_up[0] + Dx*cam_right[0]
    ray[1] = cam_loc[1] + Dy*cam_up[1] + Dx*cam_right[1]
    ray[2] = cam_loc[2] + Dy*cam_up[2]
    
    # The direction of the ray as given by the line-segment between the 
    #from cam_back to the location at which the ray leaves the screen
    ray_dir_x = ray[0] - cam_back[0]
    ray_dir_y = ray[1] - cam_back[1]
    ray_dir_z = ray[2] - cam_back[2]
    
    ray_dir = normalize((ray_dir_x,ray_dir_y,ray_dir_z))
    
    ray[3:6] = ray_dir
    
    # Smallest intersection
    ray[6] = 1e10
    
    # Ray weight
    ray[7] = 1
    
    return ray



@cuda.jit
def advance_ray(ray):
    """
    Advance a ray to its intersection point (resets t_intersect to 1e10)
    """    
    for i in range(3):
        ray[i] += ray[6]*ray[i+3]
        
    ray[6] = 1e10


@cuda.jit
def translate_ray(ray,vec):
    """
    Translate ray origin by the given vector
    """
    ray[0] += vec[0]
    ray[1] += vec[1]
    ray[2] += vec[2]


@cuda.jit
def reflect_ray(ray, normal):
    
    # Compute new ray origin
    new_orig_x = ray[0] + ray[6]*ray[3]
    new_orig_y = ray[1] + ray[6]*ray[4]
    new_orig_z = ray[2] + ray[6]*ray[5]
            
    # Compute new ray direction
    inner = ray[3] * normal[0] + \
            ray[4] * normal[1] + \
            ray[5] * normal[2]
    
    new_vel_x = ray[3] - 2*inner*normal[0]
    new_vel_y = ray[4] - 2*inner*normal[1]
    new_vel_z = ray[5] - 2*inner*normal[2]
            
    ray_reflected = (new_orig_x, new_orig_y, new_orig_z,
                     new_vel_x, new_vel_y, new_vel_z, 1e10)
    
    return ray_reflected
    
@cuda.jit
def refract_ray(ray,normal,n_i,n_t):
    
    # Compute new ray origin
    new_orig_x = ray[0] + ray[6]*ray[3]
    new_orig_y = ray[1] + ray[6]*ray[4]
    new_orig_z = ray[2] + ray[6]*ray[5]
    
    # Compute new ray direction
    inner = ray[3] * normal[0] + \
            ray[4] * normal[1] + \
            ray[5] * normal[2]
    
    r = n_i/n_t

    sqrt_arg = 1 - r**2*(1-inner**2)
    root     = sqrt(sqrt_arg)
    coef     = r*inner + root
    
    new_vel_x = r*ray[3] - coef*normal[0]
    new_vel_y = r*ray[4] - coef*normal[1]
    new_vel_z = r*ray[5] - coef*normal[2]
    
    ray_refracted = (new_orig_x, new_orig_y, new_orig_z,
                      new_vel_x, new_vel_y, new_vel_z, 1e10)
    
    return ray_refracted
    
    
    
    
@cuda.jit
def check_cube_intersect_axis(ray, axis_index, int_axis, in_cube):
    """
    Check intersection with a cube centered at (0,0,0) with side lengths 2 at a specific axis.
    
    inputs
    ------
        ray:
            Tuple of length 6 with ray origin coordinates and unit direction vector respectively
        t_intersect:
            The currently smallest known intersection time
        axis_index:
            The axis of the cube to be checked with this call; 0,1,2 means respectively x,y,z
        int_axis:
            The axis of the cube that gives the currently smallest known t_intersect
        in_cube:
            Bool that says whether it is currently thought that the origin of the ray is within the cube
    """
    
    if ray[axis_index] < -1:
        in_cube   = False
        vel_index = axis_index + 3
        
        if ray[vel_index] > 0:
            t_neg = (-1 - ray[axis_index])/ray[vel_index]
            
            if (0 <= t_neg < ray[6]):
                other_axis_1_index = (axis_index + 1) % 3
                other_vel_1_index  = other_axis_1_index + 3
                other_axis_1       = ray[other_axis_1_index] + \
                                      t_neg*ray[other_vel_1_index]
                
                if (abs(other_axis_1) < 1):
                    other_axis_2_index = (axis_index + 2) % 3
                    other_vel_2_index  = other_axis_2_index + 3
                    other_axis_2       = ray[other_axis_2_index] + \
                                          t_neg*ray[other_vel_2_index]
                                         
                    if (abs(other_axis_2) < 1):
                        ray[6]   = t_neg
                        int_axis = axis_index
                        
                        
    elif ray[axis_index] > 1:
        in_cube   = False
        vel_index = axis_index + 3
        
        if ray[vel_index] < 0:
            t_pos = (1 - ray[axis_index])/ray[vel_index]
            
            if (0 <= t_pos < ray[6]):
                other_axis_1_index = (axis_index + 1) % 3
                other_vel_1_index  = other_axis_1_index + 3
                other_axis_1       = ray[other_axis_1_index] + \
                                      t_pos*ray[other_vel_1_index]
                                     
                if (abs(other_axis_1) < 1):
                    other_axis_2_index = (axis_index + 2) % 3
                    other_vel_2_index  = other_axis_2_index + 3
                    other_axis_2       = ray[other_axis_2_index] + \
                                          t_pos*ray[other_vel_2_index]
                                         
                    if (abs(other_axis_2) < 1):
                        ray[6]   = t_pos
                        int_axis = axis_index
    
    return int_axis, in_cube



@cuda.jit
def cube_intersect(ray):
    """
    Check intersection with a cube centered at (0,0,0) with side lengths 2
    
    inputs
    ------
        ray:
            Tuple of length 6 with ray origin coordinates and unit direction vector respectively
        t_intersect:
            The currently smallest known intersection time
        ax_intersect
            The axis of the cube that gives the smallest t_intersect (-1 means no intersection)
            
            
    outputs
    -------
        in_cube:
            Bool that says whether the ray origin is within the cube
        intersect_found
            Bool that says wther an intersection with the cube was found
    
    """
    
    ax_intersect    = -1
    in_cube         = True
    
    ax_intersect, in_cube = check_cube_intersect_axis(ray, 0, ax_intersect, in_cube)
    ax_intersect, in_cube = check_cube_intersect_axis(ray, 1, ax_intersect, in_cube)
    ax_intersect, in_cube = check_cube_intersect_axis(ray, 2, ax_intersect, in_cube)

    
    return ax_intersect, in_cube  
    
    


@cuda.jit
def sphere_intersect(ray, center, R, near):
    """
    Calculate an intersection with a sphere.
    
    Inputs
    ------
        ray
            The ray for the intersection
        center 
            The center of the sphere
        R
            The radius of the sphere
        near
            Boolean saying whether the requested intersection is the near or the far one
            
    
    Outputs
    -------
        ray
            See above
        intersect_found
            Bool that says whether an intersection of the ray with the sphere was found
    """
    
    center_diff_x = ray[0] - center[0]
    center_diff_y = ray[1] - center[1]
    center_diff_z = ray[2] - center[2]
    
    inner    = center_diff_x*ray[3] + center_diff_y*ray[4] + center_diff_z*ray[5]
    normsq   = center_diff_x**2 + center_diff_y**2 + center_diff_z**2
    sqrt_arg = inner*inner + R**2 - normsq
    
    intersect_found = False
    
    if (sqrt_arg >= 0):
        
        if near:
            t_int = -inner - sqrt(sqrt_arg)
        else:
            t_int = -inner + sqrt(sqrt_arg)
        
        if (0 <= t_int < ray[6]):
            intersect_found = True
            ray[6]          = t_int
            
    return intersect_found


@cuda.jit
def get_sphere_normal(ray, center):
    
    normal_x = ray[0] + ray[6]*ray[3] - center[0]
    normal_y = ray[1] + ray[6]*ray[4] - center[1]
    normal_z = ray[2] + ray[6]*ray[5] - center[2]
    
    norm = sqrt(normal_x*normal_x + 
                normal_y*normal_y + 
                normal_z*normal_z)
    
    normal = (normal_x/norm,
              normal_y/norm,
              normal_z/norm)
    
    return normal


@cuda.jit
def get_background(ray, brightness):
    # TODO: Write function, docstring
    f_x = sin(10*np.pi*(ray[3]+1)/2)**2
    f_y = sin(10*np.pi*(ray[4]+1)/2)**2
    f_z = sin(10*np.pi*(ray[5]+1)/2)**2

    R = 0.
    G = 0.
    B = brightness*f_x*f_y*f_z

    return B, G, R


@cuda.jit
def argmin(it):
    
    Min = it[0]
    out = 0
    i_  = 0
    
    for x in it[1:]:
        
        i_ += 1
        
        if x < Min:
            Min = x
            out = i_
            
    return out







@cuda.jit
def get_kernel_rs(kernel_rs,depth_field,focus_depth,focus_dropoff):

    i,j = cuda.grid(2)

    if (i >= kernel_rs.shape[0]) or (j >= kernel_rs.shape[1]):
        return

    t_int    = depth_field[i,j]
    
    if (t_int > 1e9):
        t_int = 0
    
    L        = (abs(t_int - focus_depth) + 1e-3)*focus_dropoff
    kernel_r = ceil(L-0.5)
    
    kernel_rs[i,j] = kernel_r
    
    
@cuda.reduce
def max_reduce(a, b):
    return max(a,b)


@cuda.jit
def Defocus_kernel(canvas_in, depth_field,
                   canvas_out, kernel_rs, kernel_r_max,
                   focus_depth, focus_dropoff):
    
    i,j = cuda.grid(2)

    if (i >= kernel_rs.shape[0]) or (j >= kernel_rs.shape[1]):
        return
    
    # L = (abs(depth_field[i,j] - focus_depth) + 1e-3)*focus_dropoff
    
    for a_ in range(2*kernel_r_max+1):
        a  = a_ - kernel_r_max
        i_ = i  + a
        
        for b_ in range(2*kernel_r_max+1):
        
            b  = b_ - kernel_r_max
            j_ = j + b
            
            if not ((0 <= i_ < kernel_rs.shape[0]) and (0 <= j_ < kernel_rs.shape[1])):
                continue
            
            r = max(abs(a),abs(b))
            
            if r > kernel_rs[i_,j_]:
                continue

            L_ = (abs(depth_field[i_,j_] - focus_depth) + 1e-3)*focus_dropoff
            
            # if L < L_:
            #     continue
            
            color  = canvas_in[i_,j_]
            
            i_min = max(a-0.5,-L_)
            i_max = min(a+0.5,L_)
            
            weight_i = ((i_max-i_min)/L_ + (sin(np.pi*i_max/L_)-sin(np.pi*i_min/L_))/np.pi)/2
            
            j_min = max(b-0.5,-L_)
            j_max = min(b+0.5,L_)
            
            weight_j = ((j_max-j_min)/L_ + (sin(np.pi*j_max/L_)-sin(np.pi*j_min/L_))/np.pi)/2
            
            weight = weight_i*weight_j
            
            canvas_out[i,j,0] += weight*color[0]
            canvas_out[i,j,1] += weight*color[1]
            canvas_out[i,j,2] += weight*color[2]
            
    canvas_out[i,j,0] = min(canvas_out[i,j,0],1)
    canvas_out[i,j,1] = min(canvas_out[i,j,1],1)
    canvas_out[i,j,2] = min(canvas_out[i,j,2],1)
    
    




def Defocus(canvas_in, depth_field, focus_depth, focus_dropoff,
            blocks_per_grid, threads_per_block):
    
    kernel_rs = cuda.device_array(depth_field.shape, dtype = np.uint8)

    get_kernel_rs[blocks_per_grid,
                  threads_per_block](kernel_rs,
                                     depth_field,
                                     focus_depth,
                                     focus_dropoff)
                                     
    kernel_r_max = max_reduce(kernel_rs.ravel())
    canvas_out   = cuda.device_array_like(canvas_in)
    
    Defocus_kernel[blocks_per_grid,
                   threads_per_block](canvas_in,
                                      depth_field,
                                      canvas_out,
                                      kernel_rs,
                                      kernel_r_max,
                                      focus_depth,
                                      focus_dropoff)
    
    
    return canvas_out
                    