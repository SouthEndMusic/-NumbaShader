import numpy as np
import cv2
from numba import cuda

@cuda.jit # Just-in-time CUDA compilation; data types are inferred from first run
def test_kernel(canvas):
    
    # Get the indices of the thread; one thread per pixel of the canvas
    i,j = cuda.grid(2)
    
    # Check whether indices are within canvas size
    if (i >= canvas.shape[0]) or (j >= canvas.shape[1]):
        return
    
    # These kernels do not return anything, they only do in-place operations on arrays
    canvas[i,j,0] = int(255*i/canvas.shape[0])
    canvas[i,j,1] = int(255*j/canvas.shape[0])
    
    

# Create canvas array on the device
canvas = cuda.device_array((1000,1000,3), dtype=np.uint8)

# Thread distribution parameters, has to do with GPU architecture
threads_per_block = np.array([16,16])
blocks_per_grid   = np.ceil(canvas.shape[:2]/threads_per_block).astype(int)

# Run the kernel
# !!!: The first call to the kernel takes longer than all subsequent ones, because
#      of the just-in-time (jit) compilation
test_kernel[tuple(blocks_per_grid), 
            tuple(threads_per_block)](canvas)

# Show result (canvas array must be copied to host)
cv2.imshow("Test kernel result", canvas.copy_to_host())
cv2.waitKey(0)
cv2.destroyAllWindows()

