'''M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
'''

from __future__ import division
from subprocess import call
import ctypes

import os


def load_so(source, name, extension, path = None, precision = 'single'):
    if not os.path.exists(source + name + '.so'):
        comp_so(source, name, extension, path, precision)
    else:
        print('library '+name +' already compiled')
    #endif
    return


def comp_so(source, n_file, extension, path = None, precision = 'single' ):
    """
    source = string with the path to source folder
    n_file = string source file name without .*
    extension = string .* of sorce file 
    """

    if (n_file == 'lib_babd'):
        if os.path.exists(source + n_file + extension) and not os.path.exists(source + n_file + '.so'):
            if precision == 'single':
                if path == None:
                    print("(S) compiling " + source + n_file + extension)
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared", '-rdc=true', '-arch=compute_50', 
                                '-code=compute_50,sm_50', '-lcusparse', '-lcudadevrt', '-lcublas_device', '-lcublas', "-o",
                                source + n_file + '.so', source + n_file + extension])
                else:                                                                                                      
                    print("(S) compiling " + source + n_file + extension + " con path")
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared", '-rdc=true', '-arch=compute_50', 
                                '-code=compute_50,sm_50', '-lcusparse', '-lcublas', '-lcudadevrt', '-lcublas_device',
                                "-I", path,  "-o",
                                source + n_file + '.so',source + n_file + extension])
                #endif
            elif precision == 'double':
                if path == None:
                    print("(D) compiling " + source + n_file + extension)
                    print('nvcc', '-O3', '-Xcompiler','"-fPIC"','-shared', '-rdc=true','-arch=compute_50', 
                                '-code=compute_50,sm_50','-lcusparse', '-lcudadevrt', '-lcublas' , '-o',
                          source + n_file + '.so', source + n_file + extension)
                    err = call(['nvcc', '-O3', '-Xcompiler','"-fPIC"','-shared',
                                '-rdc=true','-arch=compute_50', 
                                '-code=compute_50,sm_50','-lcusparse', '-lcudadevrt', '-lcublas' ,'-o',
                                source + n_file + '.so', source + n_file + extension])

                else:
                    print("(D) compiling " + source + n_file + extension + " con path")
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared", '-rdc=true','-arch=compute_50', 
                                '-code=compute_50,sm_50','-lcusparse', '-lcudadevrt', '-lcublas_device', '-lcublas' ,
                                "-I", path,  "-o",
                                source + n_file + '.so',source + n_file + extension])
                #endif
            else:
                print('Errore nella scelta di precision')
            #endif        
    else:
        if os.path.exists(source + n_file + extension) and not os.path.exists(source + n_file + '.so'):
            if precision == 'single':
                if path == None:
                    print("(S) compiling " + source + n_file + extension)
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared", '-arch=compute_50', '-code=compute_50,sm_50', "-o",
                                source + n_file + '.so', source + n_file + extension])
                else:                                                                                                    
                    print("(S) compiling " + source + n_file + extension + " con path")
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared", '-arch=compute_50', '-code=compute_50,sm_50',
                                "-I", path,  "-o",\
                                source + n_file + '.so',source + n_file + extension])
                #endif
            elif precision == 'double':
                if path == None:
                    print("compiling " + source + n_file + extension)
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared",'-arch=compute_50', '-code=compute_50,sm_50', "-o",
                                source + n_file + '.so', source + n_file + extension])
                else:
                    print("compiling " + source + n_file + extension + " con path")
                    err = call(["nvcc","-Xcompiler","-fPIC","-shared",'-arch=compute_50', '-code=compute_50,sm_50', 
                                "-I", path,  "-o",\
                                source + n_file + '.so',source + n_file + extension])
                #endif
            else:
                print('Error: precision not known')
            #endif
        #endif
    #endif
    return
