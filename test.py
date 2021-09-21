'''M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
'''

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context

import scipy
from scipy.sparse import lil_matrix,csr_matrix,csc_matrix, coo_matrix, bsr_matrix, eye
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import norm
import numpy as np
import math

from ctypes import c_int, c_ulong

from timeit import default_timer
import sys

from compiler import *
from babd_auxil import *


def test(N = '9', n = '8',  nb_slices = '3'):

    N = int(N)
    n = int(n)
    nb_slices = int(nb_slices)

    tmplog2 = math.log2( nb_slices+1 )
    tmpdiv = N/nb_slices
    
    if (abs(tmpdiv - math.floor(tmpdiv))>1e-15):
        print('Error: nb_slices must divide N')
        return
    elif (abs(tmplog2 - math.floor(tmplog2))>1e-15):
        print('Error: nb_slice+1 must be a power of 2')
        #return
    elif (n>8):
        print('Error: n must be less than or equal 8')
        return
    #endif

    # Inizialize GPU context
    drv.init()
    ctx = make_default_context()

    #define BABD random test atrix
    J_babd = empty_babd(N,n)
    J_babd.data = np.random.rand((N+1)*2, 2*n,2*n)
    b_babd = np.random.rand((N+1)*2*n)

    #solve with spsolve
    J_csr = J_babd.tocsr()
    start  = default_timer()
    x_babd = scipy.sparse.linalg.spsolve(J_csr, b_babd)
    stop   = default_timer()
    print('spsolve time: ', stop - start)
    
    # initialize array to be communicated to GPU
    J_batch = babd2batched(J_babd,N,n)
    b_batch = b_babd.reshape((N+1, 1, 2*n))
    batchCount = N+1
    mm = 2*2*n
    nn = 2*n
   
    slice_size = int( N/nb_slices )
    
    
    Mdata_batch    = np.zeros((int(math.log(nb_slices+1, 2)), 1, 2*nn*nn*(nb_slices+1)))
    Mindices_batch = np.zeros((int(math.log(nb_slices+1, 2)), 1, 2*nn*nn*(nb_slices+1))) #, dtype = np.int)
    Mindptr     = 2*nn*np.arange(nn*(nb_slices+1)+1, dtype = np.int)
    
    T_batch    = np.zeros((batchCount+1, nn, nn))
    Aout_batch = np.zeros((batchCount+1, nn, mm))
    
    Tau_batch  = np.zeros((nb_slices+1, 1, nn))
    R_batch    = np.zeros(((slice_size-1)*nb_slices, nn, nn))
    
    info =  gpuarray.to_gpu(np.array([0], dtype = np.int))
    elapsed =  gpuarray.to_gpu(np.array([0.0]))
    
    J_batchT, Ain_gpu, Ain_arr = babd2gpu(J_babd.copy(),N,n)
    
    Aout_gpu = gpuarray.to_gpu(Aout_batch)
    Aout_arr = bptrs(Aout_gpu)

    T_gpu = gpuarray.to_gpu(T_batch)
    T_arr = bptrs(T_gpu)

    Tau_gpu = gpuarray.to_gpu(Tau_batch)
    Tau_arr = bptrs(Tau_gpu)
    
    R_gpu = gpuarray.to_gpu(R_batch)
    R_arr = bptrs(R_gpu)
    
    Mdata_gpu = gpuarray.to_gpu(Mdata_batch)
    Mdata_arr = bptrs(Mdata_gpu)
    
    Mindices_gpu = gpuarray.to_gpu(Mindices_batch)
    Mindices_arr = bptrs(Mindices_gpu)
    
    b_gpu = gpuarray.to_gpu(b_batch.copy())
    b_arr = bptrs(b_gpu)
    
    x_gpu = gpuarray.to_gpu(np.zeros_like(b_batch))
    x_arr = bptrs(x_gpu)
    
    
    start = default_timer()
    
    parasof_hybrid_FACT( int(mm), int(nn), 
                         int(Ain_arr.gpudata),
                         int(Aout_arr.gpudata), #Aout contiene il sistema ridotto!
                         int(T_arr.gpudata), int(R_arr.gpudata), int(Tau_arr.gpudata),
                         int(Mdata_arr.gpudata), int(Mindices_arr.gpudata), 
                         int(b_arr.gpudata), 
                         int(x_arr.gpudata), int(nb_slices), c_ulong(batchCount),
                         int(info.ptr), int(elapsed.ptr))

    stop = default_timer()
    print('parasof_FACT time:', elapsed.get()[0] )
    

    xx = x_gpu.get()
    x = xx.flatten()
    print('||x_gpu - x ||/||x|| = ', np.linalg.norm(x-x_babd)/np.linalg.norm(x_babd))
    
    
    b_gpu = gpuarray.to_gpu(b_batch.copy())
    b_arr = bptrs(b_gpu)
    x_gpu = gpuarray.to_gpu(np.zeros_like(b_batch))
    x_arr = bptrs(x_gpu)
    
    bb = b_gpu.get()
    b = bb.flatten()
    print('||b_gpu - b ||/||b|| = ', np.linalg.norm(b-b_babd)/np.linalg.norm(b_babd))
    
    start = default_timer()
    parasof_hybrid_SOLV( int(mm), int(nn), 
                         int(Ain_arr.gpudata),
                         int(Aout_arr.gpudata), #Aout contiene il sistema ridotto!
                         int(T_arr.gpudata), int(R_arr.gpudata), 
                         int(Mdata_arr.gpudata), int(Mindices_arr.gpudata), 
                         int(b_arr.gpudata), 
                         int(x_arr.gpudata), int(nb_slices), c_ulong(batchCount),
                         int(info.ptr), int(elapsed.ptr))
    stop = default_timer()
    print('parasof_SOLV time:', elapsed.get()[0] )
    
    xx = x_gpu.get()
    x = xx.flatten()
    print('||x_gpu - x ||/||x|| = ', np.linalg.norm(x-x_babd)/np.linalg.norm(x_babd))
    
    
    Ain = Ain_gpu.get()
    Aout = Aout_gpu.get()
    
    
    free([Ain_gpu, Ain_arr,
          Aout_gpu, Aout_arr,
          T_gpu, T_arr, Tau_gpu, Tau_arr, R_gpu, R_arr, 
          b_gpu, b_arr, x_gpu, x_arr, Mdata_gpu, Mdata_arr, Mindices_gpu, Mindices_arr, info, elapsed])

    ctx.pop()
    clear_context_caches()

    return


if (len(sys.argv) < 4):
    test()
else:
    test(N = sys.argv[1], n = sys.argv[2],  nb_slices = sys.argv[3])
