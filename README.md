# NumbaShader

## About

This repository contains an implementation of [shaders](https://en.wikipedia.org/wiki/Shader) using [Numba for CUDA GPUs](https://numba.pydata.org/numba-doc/latest/cuda/index.html). The 3D shaders use the concept of [ray marching](https://en.wikipedia.org/wiki/Ray_marching).

This is simply a hobby project, I make no claims about the robustness or efficiency of this code.


## Examples

__Examples of animations__
- [Reflecting and refracting sphere in Menger sponge](https://www.instagram.com/p/Cjvhu3dAv3z/)
- [Game of life in 3D with depth of field](https://www.instagram.com/p/CkBEltjgbAo/)
- [Mandelbrot bulb](https://www.instagram.com/p/CjcqTozMmcz/)
- [N body chaos](https://www.instagram.com/p/CknVHILAUXi/)

__Example stills__

<img src="https://user-images.githubusercontent.com/74617371/210435306-26c36c2c-5fda-46df-89d1-1922246bc1ac.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/74617371/210582221-d73e9965-9d0b-4ab6-8126-d92937bf4100.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/74617371/210626449-43c7d56f-5477-4b9a-91b5-0b5a72ce8435.png" width=250 height=250>


## Contents

The folder `NumbaShader3D` contains:
- The script `shader_example_general.py` which gives a short introduction on how a shader could be constructed using `numba.cuda`,
- The script `Shader_setupy.py` which contains classes for a 3D camera model and viewing the shader output interactively with OpenCV,
- The script `Shader_tools.py` which contains functions for computing the path of a ray trough a scene, based on intersections, reflections and refractions, as well as a simple method for creating depth of field.

the folder `Fractals_on_text` contains:
- The script `Poisson_2D.py` for numerically solving the Poisson equation with either homogeneous Neumann or homogeneous Dirichlet boundary conditions,
- several attempts at wrapping sections of the Mandelbrot set around letters in the notebooks `Text_to_fractal.ipynb` and `Text_to_fractal_2.ipynb`.

The main folder contains some more examples of 3D shaders:
- The script `Cube_grid.py` contains an implementation for intersecting rays with a grid of cubes defined by 3D boolean array. Furthermore there are functions to define the Menger sponge or some text in the form of such a cube grid,
- The script `Game_of_life_3D.py` also uses the cube grid implementation, but applies to the cube grid a 3D version of the [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) algorithm to create animations as seen in the second animation linked to below,
- The script `Menger_sponge.py` implements the first animation linked to below with a sphere in the Menger sponge,
- The script `Menger_sponge.py` uses the 3D shader to visualize the [Mandelbulb](https://en.wikipedia.org/wiki/Mandelbulb), see the third animation linked to below,
- The script `N_body_fractal.py` implements a shader to visualise the chaos arising from the [N body problem](https://en.wikipedia.org/wiki/N-body_problem), see the last animation linked to below.


Lastly the script `Makevid.py` contains a simple function to show and save animations as video files using OpenCV.





