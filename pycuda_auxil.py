'''M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
'''

import pycuda.gpuarray as gpuarray

import ctypes


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """
   
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=ctypes.c_void_p)

def free(device_list):
    for a in device_list:
        a.gpudata.free()
    #endfor
    return

def batched_to_gpu(x_batch):
    x_gpu = gpuarray.to_gpu(x_batch)
    x_arr = bptrs(x_gpu)
    return x_gpu, x_arr
