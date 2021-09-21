'''M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
'''

import math
import numpy as np

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
from pycuda_auxil import *

import ctypes

import scipy
from scipy.sparse import lil_matrix,csr_matrix,csc_matrix, coo_matrix, bsr_matrix, eye
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import norm

def random_system_BABD(N, n):
    '''
    genera una matrice BABD e un vettore RHS batched con entrate random con N+1 blocchi riga con blocchi di dim 2n x 2n 
    '''
    
    J_babd = empty_babd(N,n)
    J_babd.data = np.random.rand((N+1)*2, 2*n,2*n)
    b_babd = np.random.rand((N+1)*2*n)

    return J_babd, b_babd 



def read_random_system_BABD(cartella, nb_slices, N, nn):
    '''
    legge da file
    una matrice BABD e un vettore RHS batched con entrate random con N+1 blocchi riga con blocchi di dim 2n x 2n 
    '''

    n = int(nn/2)
    #read Matrix
    J_babd = empty_babd(N,n)
    fn = cartella+'MATR'+str(nb_slices)+'N'+str(N)+'n'+str(nn)
    print('loading '+fn)
    data = np.loadtxt(fn)
    J_babd.data = data.reshape((N+1)*2, nn,nn)
    #read RHS
    fn = cartella+'RHS'+str(nb_slices)+'N'+str(N)+'n'+str(nn)
    print('loading '+fn)
    b_babd  = np.loadtxt(fn)

    return J_babd, b_babd 

def csr_load(filen):
    '''
    carica matrice CSR da file npz
    '''
    f = np.load(filen)
    data    = f['data']
    indices = f['indices']
    indptr  = f['indptr']
    
    return csr_matrix((data, indices, indptr))

def system_load(filen):
    '''
    carica matrice CSR + vettore rhs + vettore soluzione da file npz
    '''
    f = np.load(filen)
    data    = f['data']
    indices = f['indices']
    indptr  = f['indptr']
    rhs = f['rhs']
    sol = f['sol']
    
    return csr_matrix((data, indices, indptr)), rhs, sol

def ababd2babd(A, N, n, p):
    '''
    Almost BABD to BABD elimina simbolicamente i moltiplicatori di Lagrange \mu dalle matrici dell'applicazione lap-time simulator.
    '''
    return scipy.sparse.vstack([A[:N*2*n,:-p],A[N*2*n+p:,:-p]])

def babd2abd(J, N, n, p, b = None, x = None):
    '''
    babd2abd trasforma la Jacobiana J del problema controllo ottimo dalla forma BABD alla forma ABD 
    permutando le equazioni e le variabili del sistema.
    NB: il riordinamento e' da applicare anche al RHS del sistema!
    '''
    J_ord = lil_matrix((N*(2*n)+(2*n+p), N*(2*n)+(2*n+p)))
    
    if (not isinstance(J, lil_matrix)):
        J = J.tolil()
    
    # H cappuccio
    J_ord[0:n,0:p] = J[N*(2*n):N*(2*n)+n, N*(2*n)+2*n:N*(2*n)+2*n+p].copy()
    
    # H_0 cappuccio
    J_ord[0:n,  p:p+2*n] = J[N*(2*n):N*(2*n)+n, 0:2*n].copy()
    J_ord[n:n+p,p:p+2*n] = J[N*(2*n)+2*n:N*(2*n)+2*n+p, 0:2*n].copy()
    
    # H_N cappuccio
    J_ord[(n+p)+N*(2*n):(n+p)+N*(2*n)+n, p+N*(2*n):p+N*(2*n)+2*n] = J[n+N*(2*n):N*(2*n)+2*n, N*(2*n):N*(2*n)+2*n].copy()
    
    # A_k^{-/+}
    for i in range(N):
        J_ord[(n+p)+i*2*n:(n+p)+(i+1)*2*n,p+i*(2*n):p+i*(2*n)+2*2*n] = J[i*2*n:(i+1)*2*n,i*(2*n):i*(2*n)+2*2*n].copy()
    #end
    
    if (x is None) and (b is None):
        #modifica al return!!!
        #return J_ord
        return J_ord[p:,p:]
    else:
        b_ord = np.zeros_like(b)
        x_ord = np.zeros_like(x)
        
        b_ord[0:n] = b[N*(2*n):N*(2*n)+n].copy()
        x_ord[0:p] = x[N*(2*n)+2*n:N*(2*n)+2*n+p].copy()
        
        b_ord[n:n+p] = b[N*(2*n)+2*n:N*(2*n)+2*n+p].copy()
        x_ord[p:p+2*n] = x[0:2*n].copy()
        
        b_ord[(n+p)+N*(2*n):(n+p)+N*(2*n)+n] = b[n+N*(2*n):N*(2*n)+2*n].copy()
        x_ord[p+N*(2*n):p+N*(2*n)+2*n] = x[N*(2*n):N*(2*n)+2*n].copy()
    
        for i in range(N):
            b_ord[(n+p)+i*2*n:(n+p)+(i+1)*2*n] = b[i*2*n:(i+1)*2*n].copy()
            x_ord[p+i*(2*n):p+i*(2*n)+2*2*n] = x[i*(2*n):i*(2*n)+2*2*n].copy()
        #end
        
        return J_ord[p:,p:], b_ord[p:], x_ord[p:]

def abd2bsr(J,N,n):
    '''
    abd2bsr converte la matrice ABD dal formato lil al formato bsr (Block Sparse Row), con blocchi di taglia nxn
    '''
    
    d = (N+1)*(2*n)

    indptr  = np.zeros((2*(N+1)+1,), dtype=int)
    indices = np.zeros((2*N*4+4,), dtype=int)
    
    indptr[0] = 0
    indptr[1] = 2
    for j in range(2,(N+1)*2):
        indptr[j] = indptr[j-1] + 4
    #end
    indptr[(N+1)*2] = indptr[(N+1)*2-1] + 2
    
    data = np.zeros((2*N*4+4,n,n), dtype=J.dtype)
    
    data[0,:,:] = J[:n,:n].toarray()
    data[1,:,:] = J[:n,n:2*n].toarray()
    indices[0] = 0
    indices[1] = 1
    for i in range(0,N):
        for k in range(2):
            for j in range(4):
                indices[2+8*i+k*4+j] = 2*i+j
                data[2+8*i+k*4+j,:,:] = J[n+(2*i+k)*n:n+(2*i+k+1)*n,(2*i+j)*n:(2*i+j+1)*n].toarray()
            #end
        #end
    #end
    
    data[2*N*4+2,:,:] = J[d-n:,d-2*n:d-n].toarray()
    data[2*N*4+3,:,:] = J[d-n:,d-n:].toarray()
    indices[2*N*4+2] = 2*(N+1)-2
    indices[2*N*4+3] = 2*(N+1)-1
        
    Jbsr = bsr_matrix((data,indices,indptr), shape = (d,d))
    
    return Jbsr

def babd2bsr(J,N,n):
    '''
    babd2bsr converte la matrice ABD dal formato lil al formato bsr (Block Sparse Row), con blocchi di taglia nxn
    '''
    
    d = J.shape[0]

    indptr  = np.zeros((N+2,), dtype=int)
    indices = np.zeros((N+1)*2, dtype=int)
    
    indptr[0] = 0
    for j in range(1,N+1):
        indptr[j] = indptr[j-1] + 2
    #end
    indptr[N+1] = N*2+2
    
    
    data = np.zeros((N*2+2,2*n,2*n), dtype=J.dtype)
    
    for i in range(0,N):
        for k in range(2):
                indices[indptr[i]+k] = i+k
                data[indptr[i]+k,:,:] = J[i*2*n:(i+1)*2*n,(i+k)*2*n:(i+k+1)*2*n].toarray()
        #end
    #end
    
    data[indptr[N],:,:]   = J[d-2*n:,:2*n].toarray()
    data[indptr[N]+1,:,:] = J[d-2*n:,-2*n:].toarray()
    indices[indptr[N]]   = 0
    indices[indptr[N]+1] = N
        
    Jbsr = bsr_matrix((data,indices,indptr), shape = (d,d))
    
    return Jbsr

def empty_babd(N,n,dtype = np.float64):
    '''
    ritorna la struttura BABD in formato BSR di una matrice di tutti zeri con N blocchi riga interni, con blocchi quadrati nxn
    '''
    d = (N+1)*(2*n)

    indptr  = np.zeros((N+2,), dtype=int)
    indices = np.zeros((N+1)*2, dtype=int)
    
    indptr[0] = 0
    for j in range(1,N+1):
        indptr[j] = indptr[j-1] + 2
    #end
    indptr[N+1] = N*2+2
    
    data = np.zeros((N*2+2,2*n,2*n), dtype=dtype)
    
    for i in range(0,N):
        for k in range(2):
                indices[indptr[i]+k] = i+k
        #end
    #end
    
    indices[indptr[N]]   = 0
    indices[indptr[N]+1] = N
        
    Jbsr = bsr_matrix((data,indices,indptr), shape = (d,d))
    
    return Jbsr

def empty_abd(N,n,dtype = np.float64):
    '''
    ritorna la struttura ABD in formato BSR di una matrice di tutti zeri con N blocchi riga interni, con blocchi quadrati nxn
    '''
    d = (N+1)*(2*n)

    indptr  = np.zeros((2*(N+1)+1,), dtype=int)
    indices = np.zeros((2*N*4+4,), dtype=int)
    
    indptr[0] = 0
    indptr[1] = 2
    for j in range(2,(N+1)*2):
        indptr[j] = indptr[j-1] + 4
    #end
    indptr[(N+1)*2] = indptr[(N+1)*2-1] + 2
    
    data = np.zeros((2*N*4+4,n,n), dtype = dtype)
    
    indices[0] = 0
    indices[1] = 1
    for i in range(0,N):
        for k in range(2):
            for j in range(4):
                indices[2+8*i+k*4+j] = 2*i+j
            #end
        #end
    #end
    
    indices[2*N*4+2] = 2*(N+1)-2
    indices[2*N*4+3] = 2*(N+1)-1
        
    Jbsr = bsr_matrix((data,indices,indptr), shape = (d,d))
    
    return Jbsr

def SOF_even_indices(n,N):
    '''
    ritorna gli indici 1-d del k-esimo blocco di un vettore dove i blocchi sono (1 x n).
    '''
    indices = [2*i*n+j for i in range(math.ceil(N/2)) for j in range(n)]
    
    return indices

def SOF_odd_indices(n,N):
    '''
    ritorna gli indici 1-d del k-esimo blocco di un vettore dove i blocchi sono (1 x n).
    '''
    indices = [(2*i+1)*n+j for i in range(math.ceil(N/2)) for j in range(n)]
    
    return indices

# Non necessaria qui
def aabd2abd(J,n,p):
    '''
    aabd2abd (Almost abd 2 abd) applica l'eliminazione di Gauss per diagonalizzare il blocco composto dalle prime p righe di J
    J e' una matrice riordinata con la funzione BABD2ABD (senza la modifica al return!!!)
    In questo modo e' possibile trascurare le prime p righe/colonne di J che diventa in forma canonica ABD
    NB: Questa procedura di eliminazione va utilizzata anche sul RHS del sistema lineare! 
    '''
    H = J_ord[0:n+p,0:p+2*n]#.toarray()
    
    for i in range(p):
        piv        = H[i,i]
        H[i,p+2]   =  H[i,p+2] - (piv/piv)*H[i,p+2]
        H[i,i+2*n] =  H[i,i+2*n] - (piv/piv)*H[i,i+2*n]
    #endfor
    H[2,(p+2)+1] =  H[2,p+2+1] - (piv/piv)*H[2,p+2+1]
    
    return H

# In realta' questa funzione non e' necessaria perche' le variabili con indici 0,...,p-1 dipendono dalle
# variabili p+2, 2*n,...,2*n+p-1, MA NON VICEVERSA -> nella risoluzione del sistema lineare le ignoro. 

def paddedBABD2bsr(J,N,n,b_babd):
    '''
    abd2bsr converte la matrice BABD dal formato lil al formato bsr (Block Sparse Row), 
    con blocchi di taglia [2*n x 2*n], previo padding.
    '''
    
    nlev = np.log2(N+1); print("nlev = ",nlev)
    Np2 = int(2**np.ceil(nlev)); print("Np2 = ",Np2)
    if Np2 > N+1:
        print("padding:")
        npad = Np2 - (N+1); print("npad = ",npad)
    else:
        npad = 0
    #endif
    b_paddedBABD = np.zeros(Np2*2*n)

    d = J.shape[0]
    print("d = ",d," , (N+1)*2*n = ",(N+1)*2*n," , n = ",n)

    indptr  = np.zeros((Np2+1,), dtype=int)
    indices = np.zeros(Np2*2, dtype=int)
    
    indptr[0] = 0
    for j in range(1,Np2):
        indptr[j] = indptr[j-1] + 2
    #end
    indptr[Np2] = Np2*2
        
    data = np.zeros((Np2*2,2*n,2*n), dtype=J.dtype)
    
    for i in range(0,Np2-1):
        for k in range(2):
            indices[indptr[i]+k] = i+k
            if i==0:
                data[indptr[i]+k,:,:] = J[i*2*n:(i+1)*2*n,(i+k)*2*n:(i+k+1)*2*n].toarray()
                b_paddedBABD[i*2*n+k*n:i*2*n+(k+1)*n] = b_babd[i*2*n+k*n:i*2*n+(k+1)*n]
            elif i>npad:
                data[indptr[i]+k,:,:] = J[(i-npad)*2*n:(i-npad+1)*2*n,(i-npad+k)*2*n:(i-npad+k+1)*2*n].toarray()
                b_paddedBABD[i*2*n+k*n:i*2*n+(k+1)*n] = b_babd[(i-npad)*2*n+k*n:(i-npad)*2*n+(k+1)*n]
            else:
                tmp1 = np.eye(2*n) #np.hstack((np.eye(2*n),-np.eye(2*n)))
                #tmp2 = np.hstack((np.zeros((n,n)),np.eye(n)))
                if k==1: tmp1 = -tmp1
                data[indptr[i]+k,:,:] = tmp1 #np.vstack((tmp1,tmp2))
                b_paddedBABD[i*2*n+k*n:i*2*n+(k+1)*n] = np.zeros(n)
            #endif
        #end
    #end
    
    data[indptr[Np2-1],:,:]   = J[d-2*n:,:2*n].toarray()
    data[indptr[Np2-1]+1,:,:] = J[d-2*n:,-2*n:].toarray()
    indices[indptr[Np2-1]]   = 0
    indices[indptr[Np2-1]+1] = Np2-1
        
    Jbsr = bsr_matrix((data,indices,indptr), shape = (Np2*2*n,Np2*2*n))
    
    return Jbsr,Np2,b_paddedBABD



def babd2gpu(J,N,n):
    '''
    babd2batched: 
    Prende in input una matrice BABD J in formato BSR i cui 
    blocchi hanno dimensione (2n x 2n) 
    e la trasforma in formato batched.
    Nel blocco 0 della matrice batched ho salvato [[B_a], [S_0]] anziche' 
    [[S_0],[B_a]] per facilitare la scrittura dell'algoritmo
    '''
    
    d = 2*n #taglia blocchi
    
    Jbatch = np.zeros((N+1,d,2*d), dtype = J.dtype)
    
    # [[B_a],
    # [S_0]]
    
    Jbatch[0,:,d:] = J.data[0].T
    Jbatch[0,:,:d] = J.data[J.indptr[N]].T
    
    for i in range(N-1):
        # [[T_i],
        # [S_{i+1}]]
        Jbatch[i+1,:,:d] = J.data[J.indptr[i]+1].T
        Jbatch[i+1,:,d:] = J.data[J.indptr[i+1]].T
    #endfor
    
    # [[T_{N-1}],
    # [B_b]]
    Jbatch[N,:,:d] = J.data[J.indptr[N-1]+1].T
    Jbatch[N,:,d:] = J.data[J.indptr[N]+1].T

    J_gpu = gpuarray.to_gpu(Jbatch)
    J_arr = bptrs(J_gpu)
    
    return Jbatch, J_gpu, J_arr

def batched2gpu(J,N,n):
    '''
    babd2batched:
    Prende in input una matrice BABD J in batched 
    e la trasferisce alla GPU in formato batched (trasponendola!!!).
    '''
    
    d = 2*n #taglia blocchi
    
    Jbatch = np.zeros((N+1,d,2*d), dtype = J.dtype)
    
    for i in range(J.shape[0]):
        Jbatch[i,:,:] = J[i,:,:].T
    #endfor

    J_gpu = gpuarray.to_gpu(Jbatch)
    J_arr = bptrs(J_gpu)
    
    return Jbatch, J_gpu, J_arr

def concatena_due_babd(J1, b1,  N1, n1, J2, b2, N2, n2):
    if (n1 != n2):
        print("ERRORE! Per concatenare due matrici BABD e' necessario che abbiamo la stessa dimensione dei blocchi interni" )
        return
    #endif
    n = n1
    N = N1+N2
    
    J = empty_babd(N,n,dtype = J1.dtype)
    
    for i in range(0,N1):
        for k in range(2):
                J.data[J.indptr[i]+k] = J1.data[J1.indptr[i]+k].copy()
        #end
    #end
    
    for i in range(N1,N1+N2):
        for k in range(2):
                J.data[J.indptr[i]+k] = J2.data[J2.indptr[i-N1]+k].copy()
        #end
    #end
        
    J.data[J.indptr[N]]   = J2.data[J2.indptr[N2]].copy()
    J.data[J.indptr[N]+1] = J2.data[J2.indptr[N2]+1].copy()

    b = np.concatenate((b1[:N1*2*n], b2))
    
    return J, b, N

def concatena_babd(Jo, bo, No, n, times):
    '''
    funzione per creare test di dimensioni grandi a partire dalle matrici Adria
    '''
    N = No
    J = Jo.copy()
    b = bo.copy()
    for i in range(times-1):
        J, b, N = concatena_due_babd(Jo, bo, No, n, J, b, N, n)
    #endfor
    J, b, N = concatena_due_babd(Jo, bo, No-1, n, J, b, N, n) 
    return J, b, N

def babd2batched(J,N,n):
    '''
    babd2batched Prende in input una matrice BABD J in formato BSR i cui 
    blocchi hanno dimensione (2n x 2n) 
    e la trasforma in formato batched.
    Nel blocco 0 della matrice batched ho salvato [[B_a], [S_0]] anziche' 
    [[S_0],[B_a]] per facilitare la scrittura dell'algoritmo
    '''
    
    d = 2*n #taglia blocchi
    
    Jbatch = np.zeros((N+1,2*d,d), dtype = J.dtype)
    
    # [[B_a],
    # [S_0]]
    Jbatch[0,d:,:] = J.data[0].copy()
    Jbatch[0,:d,:] = J.data[J.indptr[N]].copy()
    
    for i in range(N-1):
        # [[T_i],
        # [S_{i+1}]]
        Jbatch[i+1,:d,:] = J.data[J.indptr[i]+1].copy()
        Jbatch[i+1,d:,:] = J.data[J.indptr[i+1]].copy()
    #endfor
    
    # [[T_{N-1}],
    # [B_b]]
    Jbatch[N,:d,:] = J.data[J.indptr[N-1]+1].copy()
    Jbatch[N,d:,:] = J.data[J.indptr[N]+1].copy()

    return Jbatch

def batched2babd(Jbatch,N,n):
    '''
    babd2batched:
    Prende in input una matrice BABD J in formato BSR i cui 
    blocchi hanno dimensione (2n x 2n) 
    e la trasforma in formato batched.
    Nel blocco 0 della matrice batched ho salvato [[B_a], [S_0]] anziche' 
    [[S_0],[B_a]] per facilitare la scrittura dell'algoritmo
    '''
    
    d = 2*n #taglia blocchi
    
    J_babd = empty_babd(N,n,dtype = np.float64)
    
    # [[B_a],
    # [S_0]]
    J_babd.data[0] = Jbatch[0,d:,:].copy()
    J_babd.data[J_babd.indptr[N]] = Jbatch[0,:d,:].copy()
    
    for i in range(N-1):
        # [[T_i],
        # [S_{i+1}]]
        J_babd.data[J_babd.indptr[i]+1] = Jbatch[i+1,:d,:].copy()
        J_babd.data[J_babd.indptr[i+1]] = Jbatch[i+1,d:,:].copy()
    #endfor
    
    # [[T_{N-1}],
    # [B_b]]
    J_babd.data[J_babd.indptr[N-1]+1] = Jbatch[N,:d,:].copy()
    J_babd.data[J_babd.indptr[N]+1] = Jbatch[N,d:,:].copy()

    return J_babd

def batchedgpu2triangular(Jbatch,N,n):
    '''
    batchedgpu2triangular:
    Prende in input una matrice BABD J in formato BSR i cui 
    blocchi hanno dimensione (2n x 2n) 
    e la trasforma in formato batched.
    Nel blocco 0 della matrice batched ho salvato [[B_a], [S_0]] anziche' 
    [[S_0],[B_a]] per facilitare la scrittura dell'algoritmo
    '''
    
    d = 2*n #taglia blocchi

    Nsystems = int((N+1)/2)

    
    indices = np.zeros((N+1 + Nsystems,), dtype = np.int)
    data = np.zeros((N+1 + Nsystems,d,d))
    indptr  = np.hstack([2*np.arange(Nsystems+1),np.arange(2*Nsystems+1, 2*Nsystems+1+Nsystems)])

    
    for K in range(Nsystems): 
        KK = Nsystems + K
        
        
        blocco = Jbatch[K,:,:]

        data[2*K]    = np.tril(blocco[:,:d]).T
        #data[2*KK]   = blocco[:,d:]
        blocco = Jbatch[KK,:,:]
        data[2*K+1]  = blocco[:,:d].T
        
        data[N+1+K] = np.tril(blocco[:,d:]).T
        
        indices[2*K]    = K
        indices[2*K+1]  = KK
        #indices[2*KK]   = K
        indices[N+1+K] = KK
    #endfor
    
    T = bsr_matrix((data, indices, indptr))
    return T

def batchedgpu2babd(Jbatch,N,n):
    '''
    batchedgpu2triangular:
    Prende in input una matrice BABD J in formato BSR i cui 
    blocchi hanno dimensione (2n x 2n) 
    e la trasforma in formato batched.
    Nel blocco 0 della matrice batched ho salvato [[B_a], [S_0]] anziche' 
    [[S_0],[B_a]] per facilitare la scrittura dell'algoritmo
    '''
    
    d = 2*n #taglia blocchi

    J = empty_babd(N,n)

    blocco  = Jbatch[0,:,:]

    data[2*K]    = np.tril(blocco[:,:d]).T
    #data[2*KK]   = blocco[:,d:]
    blocco = Jbatch[KK,:,:]
    data[2*K+1]  = blocco[:,:d].T
    
    for K in range(1, N): 
        blocco = Jbatch[K,:,:]

        data[2*K]    = np.tril(blocco[:,:d]).T
        #data[2*KK]   = blocco[:,d:]
        blocco = Jbatch[KK,:,:]
        data[2*K+1]  = blocco[:,:d].T
        
        data[N+1+K] = np.tril(blocco[:,d:]).T
        
        indices[2*K]    = K
        indices[2*K+1]  = KK
        #indices[2*KK]   = K
        indices[N+1+K] = KK
        
    #endfor
    
    T = bsr_matrix((data, indices, indptr))
    return J

import ctypes
from compiler import *

source = './src/'
name = 'lib_babd'
extension = '.cu'   
load_so(source, name, extension, precision='double')
lib_babd_so = ctypes.CDLL(source +name + '.so')
parasof_hybrid = lib_babd_so.parasof_hybrid #versione finale ibrida ricorsiva/iterativa
parasof        = lib_babd_so.parasof        #versione ricorsiva
parasof_new    = lib_babd_so.parasof_new    #versione iterativa
parasof_new_FACT = lib_babd_so.parasof_new_FACT    #versione iterativa
iter_parasof_SOLV = lib_babd_so.iter_parasof_SOLV 
parasof_hybrid_FACT = lib_babd_so.parasof_hybrid_FACT
parasof_hybrid_SOLV = lib_babd_so.parasof_hybrid_SOLV

parasof.argtypes     = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

parasof_new.argtypes =  [ctypes.c_int, ctypes.c_int,        # int m, int n,
                         ctypes.c_void_p, ctypes.c_void_p,   # tipo **Ain, tipo **Aout,
                         ctypes.c_void_p, ctypes.c_void_p,   # tipo  **d_TArray_dev, tipo **d_TauArray_dev,
                         ctypes.c_void_p,  ctypes.c_void_p,  # tipo **bin, tipo **bout,
                         ctypes.c_void_p,                    # tipo **x,
                         ctypes.c_ulong, ctypes.c_void_p]    #  tipo_int batchCount, int *d_info

parasof_new_FACT.argtypes =  [ctypes.c_int, ctypes.c_int,        # int m, int n,
                              ctypes.c_void_p, ctypes.c_void_p,  # tipo **Ain, tipo **Aout,
                              ctypes.c_void_p, ctypes.c_void_p,  # tipo  **d_TArray_dev, tipo **d_TauArray_dev,
                              ctypes.c_void_p, ctypes.c_void_p, #tipo **csrValM, int **csrColIndM,
                              ctypes.c_void_p, ctypes.c_void_p,  # tipo **bin, tipo **bout, 
                              ctypes.c_void_p,  # tipo **x, 
                              ctypes.c_ulong,  ctypes.c_void_p]  #  tipo_int batchCount, int *d_info

iter_parasof_SOLV.argtypes = [ctypes.c_int, ctypes.c_int,        # int m, int n,
                             ctypes.c_void_p, ctypes.c_int,     #  tipo **T_array, int ldt,
                             ctypes.c_void_p, ctypes.c_void_p,  #tipo **csrValM, tipo **csrColInd,
                             ctypes.c_void_p,  # tipo **b_array,
                             ctypes.c_void_p, ctypes.c_ulong]  # tipo **x_array, tipo_int batchCount


parasof_hybrid.argtypes = [ctypes.c_int, ctypes.c_int, #int m, int n,
                           ctypes.c_void_p, ctypes.c_void_p, #tipo **Ain, tipo **Aout,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, #tipo  **d_TArray, tipo  **d_RArray, tipo **d_TauArray,
                           ctypes.c_void_p, ctypes.c_void_p, #tipo **bin, tipo **bout,
                           ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, #tipo **xin,  int nb_slices, tipo_int batchCount
                           ctypes.c_void_p, ctypes.c_void_p ] #int *d_info, tipo *d_elapsed


#parasof_hybrid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
#                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
#                           ctypes.c_void_p, ctypes.c_void_p,
#                           ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong]

parasof_hybrid_FACT.argtypes = [ctypes.c_int, ctypes.c_int, #int m, int n,
                                ctypes.c_void_p, ctypes.c_void_p, #tipo **Ain, tipo **Aout,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, #tipo  **T, tipo  **R, tipo **Tau,
                                ctypes.c_void_p, ctypes.c_void_p,  #tipo **csrValM, tipo **csrColInd,
                                ctypes.c_void_p, #tipo **b, 
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, #tipo **x,  int nb_slices, tipo_int batchCount
                                ctypes.c_void_p, ctypes.c_void_p ] #int *d_info, tipo *d_elapsed


parasof_hybrid_SOLV.argtypes = [ctypes.c_int, ctypes.c_int, #int m, int n,
                                ctypes.c_void_p, ctypes.c_void_p, #tipo **Ain, tipo **Aout,
                                ctypes.c_void_p, ctypes.c_void_p,  #tipo  **T, tipo  **R,
                                ctypes.c_void_p, ctypes.c_void_p,  #tipo **csrValM, tipo **csrColInd,
                                ctypes.c_void_p, #tipo **b, 
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, #tipo **x,  int nb_slices, tipo_int batchCount
                                ctypes.c_void_p, ctypes.c_void_p ] #int *d_info, tipo *d_elapsed


