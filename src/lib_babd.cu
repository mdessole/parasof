/* M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
//#include <cublas_v2.h>
//#include <cusparse_v2.h>
#include <cublas.h>
#include <cusparse.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#include "get_config.cu"
#include "gemm_batched.cu"
#include "geqr2.cu"
#include "trsv_device.cu"


typedef double    tipo;
typedef unsigned long int tipo_int;

#define DOUBLE_PRECISION
#define WARP_SIZE 32
#define BLOCK_SIZE WARP_SIZE
#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )
extern __shared__ tipo zdata[];

// variabili globali
int ntcol_gemm, ntcol_geqr2, nblocks_gemm, nblocks_geqr2;
dim3 blocks_gemm(1, 1, 1);
dim3 blocks_geqr2(1, 1, 1);
dim3 threads_gemm(1, 1, 1);
dim3 threads_geqr2(1, 1, 1);
dim3 threads_rec(1, 1, 1);

tipo one = 1.0, oneopp = -1.0, zero = 0.0;
__device__ tipo d_one = 1.0, d_oneopp = -1.0, d_zero = 0.0;

cusparseHandle_t cusparseHandle;
cublasHandle_t cublasHandle;
__device__ cublasHandle_t cublasHandle_dev;

cudaError_t cudaStat1 = cudaSuccess, cudaStat2 = cudaSuccess;
cusparseStatus_t cusparseStatus1 = CUSPARSE_STATUS_SUCCESS, cusparseStatus2 = CUSPARSE_STATUS_SUCCESS, cusparseStatus3 = CUSPARSE_STATUS_SUCCESS;

cublasStatus_t cublasStatus1 = CUBLAS_STATUS_SUCCESS, cublasStatus2 = CUBLAS_STATUS_SUCCESS, cublasStatus3 = CUBLAS_STATUS_SUCCESS;
__device__ cublasStatus_t d_cublasStatus1 = CUBLAS_STATUS_SUCCESS;

cusparseOperation_t cusparse_trans   = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseOperation_t cusparse_trans_t = CUSPARSE_OPERATION_TRANSPOSE;

cublasOperation_t cublas_trans   = CUBLAS_OP_N;
cublasOperation_t cublas_trans_t = CUBLAS_OP_T;

__device__ cublasOperation_t d_cublas_trans   = CUBLAS_OP_N;
__device__ cublasOperation_t d_cublas_trans_t = CUBLAS_OP_T;


/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str= cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}

#define cucheck_dev(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	assert(0);																												\
	}\
}

__device__ void cublascheck_dev(cublasStatus_t error){
switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    printf("CUBLAS_STATUS_SUCCESS %d \n", error);
    
  case CUBLAS_STATUS_NOT_INITIALIZED:
    printf("CUBLAS_STATUS_NOT_INITIALIZED  %d \n", error);
    
  case CUBLAS_STATUS_ALLOC_FAILED:
    printf("CUBLAS_STATUS_ALLOC_FAILED  %d \n", error);

  case CUBLAS_STATUS_INVALID_VALUE:
    printf("CUBLAS_STATUS_INVALID_VALUE %d \n", error);
    
  case CUBLAS_STATUS_ARCH_MISMATCH:
    printf("CUBLAS_STATUS_ARCH_MISMATCH %d \n", error);

  case CUBLAS_STATUS_MAPPING_ERROR:
   printf("CUBLAS_STATUS_MAPPING_ERROR %d \n", error);
    
  case CUBLAS_STATUS_EXECUTION_FAILED:
   printf("CUBLAS_STATUS_EXECUTION_FAILED %d \n", error);
   
  case CUBLAS_STATUS_INTERNAL_ERROR:
    printf("CUBLAS_STATUS_INTERNAL_ERROR  %d \n", error);
  default:
    printf("Unknown error \n");
  }

return;
}

__global__ void permuta_idx(tipo **Ain, tipo **Aout, tipo_int new_batchCount){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tx >= (new_batchCount*2))
    return;  
  int i;

  if (tx < new_batchCount)
    i = tx*2;
  else
    i = 2*(tx-new_batchCount)+1;

  Aout[tx] = Ain[i];
  return;
}


void compactWY_batched(int m, int n,
		       tipo **A_array, int lda, int ai, int aj,
		       tipo **Aout_array, int ldaout, int ri, int rj,
		       tipo **T_array, int ldt,
		       tipo **Tau_array,
		       int salva_R, int inverti_blocchi,
		       tipo_int batchCount, tipo_int offset){
  /* Funzione che calcola la rappresentazione WY della QR di un array batched
     nella fase di forward-reduction della versione iterativa di PARASOF.
   */ 
  
  dim3 blocks(1, 1, batchCount);
  dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2d(n, n, 1);
  size_t shmem;
  
  // step 1. Batched QR
  shmem =  sizeof(tipo)*(ntcol_geqr2*(2*BLOCK_SIZE + 2 + lda*n));//sizeof(tipo)*(batchCountperBlock*(2*(m + 1) + m*n));
  geqr2_kernel_batched<<<blocks_geqr2, threads_geqr2, shmem>>> (m, n, A_array, lda, ai, aj, Aout_array, ldaout, ri, rj, Tau_array, salva_R, inverti_blocchi, batchCount, offset);
  
  
  if (m != n){
    // ATTENZIONE: CUBLAS con il dynamic parallelism NON funziona, percio' chiamo le mie funzioni
    shmem = sizeof(tipo)*(n*SLDA(n)*2);
    parasof_That_gemm_batched_kernel<<<blocks, threads2d, shmem>>>(int(n),
									   A_array, int(lda), 
									   T_array, int(ldt),
									   batchCount);
  }else{
    shmem = sizeof(tipo)*(n*SLDA(n)*2);
    gemm_batched_smallsq_kernel<<<blocks, threads2d, shmem>>>(int(1), int(0),
								      int(n),
								      tipo(1.0),
								      A_array, ai, aj, int(lda),
								      A_array, ai, aj, int(lda),
								      tipo(0.0),
								      T_array, 0, 0, int(ldt),
								      batchCount);
  }
  
  // step 2b. calcola T aggiornando le colonne di T cappuccio
  shmem = sizeof(tipo)*(n*n);
  dlarft_dtrmv_sm32x32_kernel_batched_mio<<< blocks, threads1d, shmem >>> (n, Tau_array, T_array, ldt, batchCount);
  
  return;
}

__device__
void device_compactWY_batched(int m, int n,
			      tipo **A_array, int lda, int ai, int aj,
			      tipo **Aout_array, int ldaout, int ri, int rj,
			      tipo **T_array, int ldt,
			      tipo **Tau_array,
			      int salva_R, int inverti_blocchi,
			      tipo_int batchCount, tipo_int offset){
  /* Funzione device che calcola la rappresentazione WY della QR di un array batched
     nella fase di forward-reduction della versione ricorsiva di PARASOF.
  */ 
  int batchCountperBlock = 2;
  dim3 blocks3d(1, 1, int(batchCount/batchCountperBlock));
  dim3 threads3d(BLOCK_SIZE, 1, batchCountperBlock); // (m,1,batchCountperBlock)
  
  dim3 blocks(1, 1, batchCount);
  dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2d(n, n, 1);
  size_t shmem;
  
  // step 1. Batched QR
  shmem =  sizeof(tipo)*(batchCountperBlock*(2*BLOCK_SIZE + 2 + lda*n));//sizeof(tipo)*(batchCountperBlock*(2*(m + 1) + m*n));
  geqr2_kernel_batched<<<blocks3d, threads3d, shmem>>> (m, n, A_array, lda, ai, aj, Aout_array, ldaout, ri, rj, Tau_array, salva_R, inverti_blocchi, batchCount, offset);
  
  
  if (m != n){
    // ATTENZIONE: CUBLAS con il dynamic parallelism NON funziona, percio' chiamo le mie funzioni
    shmem = sizeof(tipo)*(batchCountperBlock*n*SLDA(n)*2);
    dim3 threads3d_2(n, n, batchCountperBlock);
    parasof_That_gemm_batched_kernel<<<blocks3d, threads3d_2, shmem>>>(int(n),
								    A_array, int(lda), 
								    T_array, int(ldt),
								    batchCount);
  }else{
    shmem = sizeof(tipo)*(n*SLDA(n)*2);
    gemm_batched_smallsq_kernel<<<blocks, threads2d, shmem>>>(int(1), int(0),
							       int(n),
							       tipo(1.0),
							       A_array, ai, aj, int(lda),
							       A_array, ai, aj, int(lda),
							       tipo(0.0),
							       T_array, 0, 0, int(ldt),
							       batchCount);
  }
  
  // step 2b. calcola T aggiornando le colonne di T cappuccio
  shmem = sizeof(tipo)*(n*n);
  dlarft_dtrmv_sm32x32_kernel_batched_mio<<< blocks, threads1d, shmem >>> (n, Tau_array, T_array, ldt, batchCount);
  
  return;
}


void sof_compactWY_batched(int m, int n,
			   tipo **A_array, int lda, int ai, int aj,
			   tipo **V_array, int ldv, int vi, int vj,
			   tipo **R_array, int ldr, int ri, int rj,
			   tipo **T_array, int ldt,
			   tipo **Tau_array,
			   int nb_slices,
			   int slice_size,
			   int offset,
			   tipo_int batchCount){
  /* Funzione che calcola la rappresentazione WY della QR di un array batched
     nella fase di forward-reduction della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa).
   */ 
  int batchCountperBlock = 1; 
  dim3 blocks3d(1, 1, int(nb_slices/batchCountperBlock));
  dim3 threads3d(BLOCK_SIZE, 1, batchCountperBlock); // (m,1,batchCountperBlock)
  
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2(n, n, 1);
  size_t shmem;


  // step 1. Batched QR
  shmem =  sizeof(tipo)*(batchCountperBlock*(2*BLOCK_SIZE + 2 + lda*n));//sizeof(tipo)*(batchCountperBlock*(2*(m + 1) + m*n));
  sof_geqr2_kernel_batched<<<blocks3d, threads3d, shmem>>>(m, n,
							   A_array, ldv, vi, vj,
							   V_array, ldv, vi, vj,
							   R_array, ldr, ri, rj,
							   Tau_array, 
							   nb_slices,
							   slice_size,
							   offset,
							   batchCount);
  
  cudaDeviceSynchronize();
  
  if (m != n){
    // ATTENZIONE: CUBLAS con il dynamic parallelism NON funziona, percio' chiamo le mie funzioni
    shmem = sizeof(tipo)*(n*SLDA(n)*2);
    sof_That_gemm_batched_kernel<<<blocks, threads2, shmem>>>(int(n),
							      V_array, int(lda), 
							      T_array, int(ldt), 
							      nb_slices,
							      slice_size,
							      offset,
							      batchCount);
  }
  
  // step 2b. calcola T aggiornando le colonne di T cappuccio
  shmem = sizeof(tipo)*(n*n);
  dlarft_dtrmv_sm32x32_kernel_batched_mio<<< blocks, threads, shmem >>> (n, Tau_array, T_array, ldt, nb_slices);

  return;

}

void sof_compactWY_batched_FACT(int m, int n,
				tipo **A_array, int lda, int ai, int aj,
				tipo **V_array, int ldv, int vi, int vj,
				tipo **R_array, int ldr, int ri, int rj,
				tipo **T_array, int ldt,
				tipo **Tau_array,
				int nb_slices,
				int slice_size,
				int offset, // iterazione di Forward Reduction (parte da 1 non da 0)
				tipo_int batchCount){
  /* Funzione che calcola la rappresentazione WY della QR di un array batched
     nella fase di forward-reduction della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa).
   */ 
  int batchCountperBlock = 1; 
  dim3 blocks3d(1, 1, int(nb_slices/batchCountperBlock));
  dim3 threads3d(BLOCK_SIZE, 1, batchCountperBlock); // (m,1,batchCountperBlock)
  
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2(n, n, 1);
  size_t shmem;


  // step 1. Batched QR
  shmem =  sizeof(tipo)*(batchCountperBlock*(2*BLOCK_SIZE + 2 + lda*n));//sizeof(tipo)*(batchCountperBlock*(2*(m + 1) + m*n));
  sof_geqr2_kernel_batched_FACT<<<blocks3d, threads3d, shmem>>>(m, n,
								A_array, ldv, vi, vj,
								V_array, ldv, vi, vj,
								R_array, ldr, ri, rj,
								Tau_array, 
								nb_slices,
								slice_size,
								offset,
								batchCount);
  
  cudaDeviceSynchronize();
  
  if (m != n){
    // ATTENZIONE: CUBLAS con il dynamic parallelism NON funziona, percio' chiamo le mie funzioni
    shmem = sizeof(tipo)*(n*SLDA(n)*2);
    sof_That_gemm_batched_kernel_FACT<<<blocks, threads2, shmem>>>(int(n),
  								   V_array, int(lda), 
  								   T_array, int(ldt), 
  								   nb_slices,
  								   slice_size,
  								   offset,
  								   batchCount);
  }
  
  // step 2b. calcola T aggiornando le colonne di T cappuccio
  shmem = sizeof(tipo)*(n*n);
  sof_dlarft_dtrmv_sm32x32_kernel_batched_FACT<<< blocks, threads, shmem >>> (n, Tau_array, T_array, ldt, nb_slices,
  									      slice_size,
  									      offset,
  									      batchCount);
  
  return;

}


__global__ void copy_square_matrix(int n, tipo **A, int lda, tipo **B, int ldb, tipo_int offset){
  /*kernel che esegue la copia di un array batched (= zone di memoria indicate dal doppio puntatore) dove gli array puntati sono matrici quadrate. 
    Utilizzato nella risoluzione dei sistemi lineari 2x2 */

  int tx = threadIdx.x;
  int batchid = blockIdx.x+offset;

  if (tx >= n)
    return;

  for (int i=0; i<n; i++)
    B[batchid][tx + i * ldb] = A[batchid][tx + i * lda];
  return;
}


__device__ void copy_batched_device(int m, int n, tipo *A, int lda, tipo *B, int ldb){
  /*funzione device che esegue la copia di un array (= zone di memoria indicate da un puntatore) */
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  if ((tx >= m) || (ty >= n))
    return;

  B[tx + ty * ldb] = A[tx + ty * lda];
  
}

__global__ void copy_batched_kernel(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout, tipo_int batchCount){
  /*kernel che esegue la copia di un array batched (= zone di memoria indicate dal doppio puntatore) */
 
  const int tz = threadIdx.z;
  tipo_int batchid = blockIdx.z * blockDim.z + tz;

  tipo *dA, *dB;

  if (batchid >= batchCount)
    return;
  
  dA = Ain_array[batchid];
  dB = Aout_array[batchid];
  
  copy_batched_device(m, n, dA, ldain, dB, ldaout);
 
  return;
}


__global__ void sof_selectXarrays_doubleptr_kernel(tipo **xin, tipo **xout, int nb_slices, int slice_size, tipo_int batchCount){
  /*kernel che costruisce il vettore di puntatori alle variabili coinvolte nel sistema ridotto dopo la fase di forward reduction nella versione ibrida*/
 
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  
  int batchid = tx*slice_size;
  
  if ( (tx < (nb_slices + 1)) && (batchid < batchCount)){
    xout[tx] = xin[batchid];
  }
  
  return;
}

__global__ void sof_selectXarrays_doubleptr_kernel_FACT(tipo **Ain, tipo **Ain_red, tipo **Aout, tipo **Aout_red, tipo **T, tipo **T_red,
							tipo **b, tipo **b_red, tipo **x, tipo **x_red,
							int nb_slices, int slice_size, tipo_int batchCount){
  /*kernel che costruisce il vettore di puntatori alle variabili coinvolte nel sistema ridotto dopo la fase di forward reduction nella versione ibrida*/
  
  int tx = blockIdx.x * blockDim.x + threadIdx.x;  
  int batchid = tx*slice_size;
  
  if ( (tx < (nb_slices + 1)) && (batchid < batchCount)){
    x_red[tx]    = x[batchid];
    T_red[tx]    = T[batchid];
    Ain_red[tx]  = Ain[batchid];
    Aout_red[tx] = Aout[batchid];
    
    // i b vanno selezionati in maniera diversa
    if ((tx<nb_slices) && (batchid + slice_size -1 <batchCount))
      b_red[tx]    = b[batchid + slice_size -1 ];
    else
      b_red[tx]    = b[batchid];
  }
  
  // i b vanno selezionati in maniera diversa
  
  return;
}


__global__ void sof_copy_batched_kernel(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout, int nb_slices, int slice_size, int offset,  tipo_int batchCount){
  int sliceId = blockIdx.z * blockDim.z + threadIdx.z;
  tipo_int batchId = sliceId * slice_size + offset;

  /*NON viene chiamata da nessuna parte, probabilmente e' inutile*/
  
  tipo *dA, *dB;

  if ((sliceId >= nb_slices) || (batchId >= batchCount))
    return;
  
  dA = Ain_array[batchId];
  dB = Aout_array[sliceId];
  
  copy_batched_device(m, n, dA, ldain, dB, ldaout);

  
  return;
}

__global__ void sof_assembly_rd_babd_batched_kernel(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout,
						    tipo **bin_array, tipo **bout_array,						    
						    int nb_slices, int slice_size, tipo_int batchCount){
  /*kernel che copia il sistema ridotto in una nuova zona di memoria (puntata da Aout_array e bout_array)
    dopo la fase di forward reduction nella versione ibrida*/
  
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.z;  
  tipo *dA, *dB;

  if (bx >= (nb_slices+1) )
    return;

  
  if (bx == nb_slices){
    /*primo blocco colonna del sistema ridotto da salvare come [[S_0],[B_a]]*/
    dA = Ain_array[bx*slice_size];
    dB = Aout_array[bx];
    copy_batched_device(m, n, dA, ldain, dB, ldaout);

    if ((tx == 0) && (ty<n))
      bout_array[bx][ty] = bin_array[bx*slice_size][ty]; 
    
  }
  else{
    /*blocchi colonna standard del sistema ridotto da salvare come [[T_i],[S_{i+1}]]*/
    dA = Ain_array[(bx+1)*slice_size-1] + n; //S'
    dB = Aout_array[bx] + n;
    copy_batched_device(n, n, dA, ldain, dB, ldaout);
    
    dA = Ain_array[bx*slice_size]; //T'
    dB = Aout_array[bx];
    copy_batched_device(n, n, dA, ldain, dB, ldaout);
    
    if ((tx == 0) && (ty<n))
      bout_array[bx][ty] = bin_array[(bx+1)*slice_size-1][ty];

  }
  
  return;
}

__global__ void copy_doubleptr_kernel(tipo **Ain, tipo **Aout, tipo_int batchCount){
  /* kernel che esegue in parallelo una copia di doppio puntatore di tipo double */
  int tx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tx<batchCount)
    Aout[tx] = Ain[tx];
  return;
}

void geqr2_batched_solver(int m, int n,
			  tipo **Ain_array, int ldain,
			  tipo **Aout_array, int ldaout,
			  tipo **T_array, int ldt, tipo **Tau_array,
			  tipo **bin_array, tipo **x_array,
			  tipo_int batchCount, tipo_int offset){
  /*Funzione device GPUCPU che risolve i sistemi 2x2 dell'ultimo step dopo la forward-reduction di PARASOF 
    Viene chiamata solo nella versione ricorsiva GPU, nella versione iterativa e nella versione ibrida */  
  dim3 block(offset, 1, 1);
  dim3 threads  (n, 1, 1);
  dim3 threads2 (n, n, 1);
  
  // QR del 1o blocco rettangolare
  // calcola V in Ain e R in Aout
  compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 1, 1, offset, offset);
  
  size_t shmem  = ( SLDA(n)*n*3 + 2*n ) * sizeof(tipo);
  // applicazione di Q = (I-VTV^T) ad Ain_array[1] e bin
  larfb_step1_last_call_kernel<<<block, threads2, shmem>>>(m, n, Ain_array, ldain, T_array, ldt, bin_array, batchCount, offset);
  
  //copio in Aout_array
  copy_square_matrix<<<block, threads>>>(n, Ain_array, ldain, Aout_array, ldaout, offset);
  
  // QR del 2o blocco quadrato (in basso a dx)
  // calcola V in Ain e R in Aout
  compactWY_batched(n, n, Ain_array+offset, ldain, n, 0, Aout_array+offset, ldaout, n, 0, T_array+offset, ldt, Tau_array+1, 1, 0, offset, offset);
  
  shmem  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
  // aggiornamento rhs (applica Q a b[1])
  larfb_step2_last_call_kernel<<<block, threads, shmem>>>(n, Ain_array, ldain, n, 0, T_array, ldt, bin_array, offset);
  
  shmem = sizeof(tipo)*n;
  dtrsv_last_call_kernel<<<block, threads, shmem>>>(n,  Aout_array, ldaout, bin_array, x_array, offset);
  
  return;
}

void geqr2_batched_solver_FACT(int m, int n,
			       tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout,
			       tipo **T_array, int ldt, tipo **Tau_array,
			       tipo **bin_array,
			       tipo **csrValM, tipo **csrColIndM, int it,
			       tipo **x_array, tipo_int batchCount, tipo_int offset){
  /*Funzione device GPUCPU che risolve i sistemi 2x2 dell'ultimo step dopo la forward-reduction di PARASOF 
    Viene chiamata solo nella versione ricorsiva GPU, nella versione iterativa e nella versione ibrida */  
  dim3 block(offset, 1, 1);
  dim3 threads  (n, 1, 1);
  dim3 threads2 (n, n, 1);
  
  // QR del 1o blocco rettangolare
  // calcola V in Ain e R in Aout
  compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 1, 1, offset, offset);
  
  size_t shmem  = ( SLDA(n)*n*3 + 2*n ) * sizeof(tipo);
  // applicazione di Q = (I-VTV^T) ad Ain_array[1] e bin
  larfb_calcolaM_step1_last_call_kernel<<<block, threads2, shmem>>>(m, n, Ain_array, ldain, T_array, ldt, csrValM, csrColIndM, it, bin_array, batchCount, offset);
  
  //copio in Aout_array
  copy_square_matrix<<<block, threads>>>(n, Ain_array, ldain, Aout_array, ldaout, offset);
  
  // QR del 2o blocco quadrato (in basso a dx)
  // calcola V in Ain e R in Aout
  compactWY_batched(n, n, Ain_array+offset, ldain, n, 0, Aout_array+offset, ldaout, n, 0, T_array+offset, ldt, Tau_array+1, 1, 0, offset, offset);


  shmem  = ( SLDA(n)*n*3 + n ) * sizeof(tipo); 
  // aggiornamento rhs (applica Q a b[1])
  larfb_calcolaM_step2_last_call_kernel<<<block, threads2, shmem>>>(n,
								    Ain_array, ldain,
								    T_array, ldt,
								    csrValM, csrColIndM, it,
								    bin_array, batchCount, offset);
  
  shmem = sizeof(tipo)*n;
  dtrsv_last_call_kernel<<<block, threads, shmem>>>(n,  Aout_array, ldaout, bin_array, x_array, offset);
  
  return;
}

__device__ void device_geqr2_batched_solver(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout, tipo **T_array, int ldt,
					      tipo **Tau_array, tipo **bin_array, tipo **x_array, tipo_int batchCount, tipo_int offset){


  /*Funzione device GPU che risolve il sistema 2x2 dell'ultimo stepdopo la forward-reduction di PARASOF 
    Viene chiamata solo nella versione con Dynamic Parallelism
    NB: Un solo thread esegue questa funzione device*/

  dim3 block(offset, 1, 1);
  dim3 threads  (n, 1, 1);
  dim3 threads2 (n, n, 1);
  
  // QR del 1o blocco rettangolare
  // calcola V in Ain e R in Aout
  device_compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 1, 1, offset, offset);
  
  size_t shmem  = ( SLDA(n)*n*3 + 2*n ) * sizeof(tipo);
  // applicazione di Q = (I-VTV^T) ad Ain_array[1] e bin
  larfb_step1_last_call_kernel<<<block, threads2, shmem>>>(m, n, Ain_array, ldain, T_array, ldt, bin_array, batchCount, offset);
  
  //copio in Aout_array
  copy_square_matrix<<<block, threads>>>(n, Ain_array, ldain, Aout_array, ldaout, offset);
  
  // QR del 2o blocco quadrato (in basso a dx)
  // calcola V in Ain e R in Aout
  device_compactWY_batched(n, n, Ain_array+offset, ldain, n, 0, Aout_array+offset, ldaout, n, 0, T_array+offset, ldt, Tau_array+1, 1, 0, offset, offset);
  
  shmem  = ( SLDA(n)*n + 2*n ) * sizeof(tipo);
  
  // aggiornamento rhs (applica Q a b[1])
  larfb_step2_last_call_kernel<<<block, threads, shmem>>>(n, Ain_array, ldain, n, 0, T_array, ldt, bin_array, offset);
  
  shmem = sizeof(tipo)*n;
  dtrsv_last_call_kernel<<<block, threads, shmem>>>(n,  Aout_array, ldaout, bin_array, x_array, offset);
  
  return;
}


__global__
void cdp_parasof(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout, tipo **T_array, int ldt,
		 tipo **Tau_array, tipo **bin_array, tipo **bout_array, tipo **x_array, tipo **ptr_tmp,
		 tipo_int batchCount, int depth, int* info){
  
  /*versione ricorsiva di PARASOF con chiamate ai kernel gestite dalla GPU
    NB: Un solo thread esegue questo kernel*/
  if ((batchCount % 2) != 0){
    *info = -1;
    return;
  }
  
  // step 0: if batchsize == 2 then solve directly
  if (batchCount == 2){    
    device_geqr2_batched_solver(m, n, Ain_array, ldain, Aout_array, ldaout, T_array, ldt,
			 Tau_array, bin_array, x_array, batchCount, int(1));
    return;
  }
  else{
    dim3 blocks(1, 1, batchCount);
    dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
    dim3 threads2d (n, n, 1);
    dim3 threads2d_rec (m, n, 1);
    
    // Copio la matrice in Out
    copy_batched_kernel<<<blocks, threads2d_rec>>>(m, n, Ain_array, ldain, Aout_array, ldaout,  batchCount);
    copy_batched_kernel<<<blocks, threads1d     >>>(n, 1, bin_array, int(1), bout_array, int(1), batchCount);
    
    // step 1. Batched compact WY representation of QR
    device_compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 0, 0, batchCount, int(1));
    
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    size_t shmem  = ( SLDA(n)*n*3 ) * sizeof(tipo); 
    //dim3 blocks2(1, 1, 1);
    parasof_larfb_gemm_batched_smallsq_kernel<<<blocks, threads2d , shmem >>>(m, n,
									     Ain_array, ldain, 
									     T_array, ldt, 
									     Aout_array, ldaout,
									     batchCount, int(1));
    // step 3. applicazione di Q = (I-VTV^T) a blocchi al rhs
    shmem  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
    parasof_larfb_gemv_batched_smallsq_kernel<<<blocks, threads2d , shmem >>>(m, n,
									     Ain_array, ldain, 
									     T_array, ldt, 
									     bin_array, bout_array,
									     batchCount, int(1));

    
    // step 4. Permutazione dei puntatori
    tipo_int new_batchCount = batchCount/2;
    dim3 blocks2 ((int) (batchCount/WARP_SIZE +1), 1, 1);

    // Copio Aout in ptr_tmp
    copy_doubleptr_kernel<<< blocks2, threads1d >>>(Aout_array, ptr_tmp, batchCount);

    // Uso ptr_tmp come input per permutare Aout 
    permuta_idx<<< blocks2, threads1d >>>(ptr_tmp, Aout_array, new_batchCount); //(int lda, int n, tipo **Ain, tipo **Aout, int new_batchCount)

    // Copio x in ptr_tmp
    copy_doubleptr_kernel<<< blocks2, threads1d >>>(x_array, ptr_tmp, batchCount);

    // Uso ptr_tmp come input per permutare x 
    permuta_idx<<< blocks2, threads1d >>>(ptr_tmp, x_array, new_batchCount); //(int lda, int n, tipo **Ain, tipo **Aout, int new_batchCount)
    
    // Copio bout in ptr_tmp
    copy_doubleptr_kernel<<< blocks2, threads1d >>>(bout_array, ptr_tmp, batchCount);
    
    // Uso ptr_tmp come input per permutare bout 
    permuta_idx<<< blocks2, threads1d >>>(ptr_tmp, bout_array, new_batchCount);

    //step 5: Ricorsione
    if (1){
    cudaStream_t s1, s2;

    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

    cdp_parasof<<<1, 1>>>(m, n,
			 Aout_array, ldaout,
			 Ain_array,  ldain,
			 T_array,    ldt,
			 Tau_array,
			 bout_array,
			 bin_array,
			 x_array,
			 ptr_tmp,
			 new_batchCount, depth+1, info);
    cudaStreamDestroy(s1);
    
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cdp_parasof<<<1, 1>>>(m, n,
			 Aout_array + new_batchCount, ldaout,
			 Ain_array  + new_batchCount, ldain,
			 T_array    + new_batchCount, ldt,
			 Tau_array  + new_batchCount,
			 bout_array + new_batchCount,
			 bin_array  + new_batchCount,
			 x_array + new_batchCount,
			 ptr_tmp + new_batchCount,
			 new_batchCount, depth+1, info);
 
    cudaStreamDestroy(s2);
    }
  }
  
  return;
}


void rec_parasof(int m, int n, tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout, tipo **T_array, int ldt,
		 tipo **Tau_array, tipo **bin_array, tipo **bout_array, tipo **x_array, tipo **ptr_tmp,
		 tipo_int batchCount, int depth, int* info){
  /*versione ricorsiva di PARASOF con chiamate ai kernel gestite dalla CPU*/
  if ((batchCount % 2) != 0){
    *info = -1;
    return;
  }
  
  // step 0: if batchsize == 2 then solve directly
  if (batchCount == 2){
    geqr2_batched_solver(m, n, Ain_array, ldain, Aout_array, ldaout, T_array, ldt,
			 Tau_array, bin_array, x_array, batchCount, int(1));
    return;
  }
  else{

    dim3 blocks(1, 1, batchCount);
    dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
    dim3 threads2d (n, n, 1);
    dim3 threads2d_rec (m, n, 1);
    
    // Copio la matrice in Out
    copy_batched_kernel<<<blocks, threads2d_rec>>>(m, n, Ain_array, ldain, Aout_array, ldaout, batchCount);
    copy_batched_kernel<<<blocks, threads1d    >>>(n, 1, bin_array, int(1), bout_array, int(1), batchCount);
   //  cudaDeviceSynchronize();
    
    // step 1. Batched compact WY representation of QR
    compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 0, 0, batchCount, int(1));
   //  cudaDeviceSynchronize();
    
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    size_t shmem  = ( SLDA(n)*n*3 ) * sizeof(tipo); 
    //dim3 blocks2(1, 1, 1);
    parasof_larfb_gemm_batched_smallsq_kernel<<<blocks, threads2d , shmem >>>(m, n,
									     Ain_array, ldain, 
									     T_array, ldt, 
									     Aout_array, ldaout, 
									     batchCount, int(1));
    // step 3. applicazione di Q = (I-VTV^T) a blocchi al rhs
    shmem  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
    parasof_larfb_gemv_batched_smallsq_kernel<<<blocks, threads2d , shmem >>>(m, n,
									       Ain_array, ldain, 
									       T_array, ldt, 
									       bin_array, bout_array,
									       batchCount, int(1));
    if (1){
      // step 4. Permutazione dei puntatori
      tipo_int new_batchCount = batchCount/2;
      
      //dim3 threads3 (WARP_SIZE, 1, 1);
      dim3 blocks3 ((int) (batchCount/WARP_SIZE +1), 1, 1);

      // Copio Aout in ptr_tmp
      copy_doubleptr_kernel<<< blocks3, threads1d >>>(Aout_array, ptr_tmp, batchCount);
      
      // Uso ptr_tmp come input per permutare Aout 
      permuta_idx<<< blocks3, threads1d >>>(ptr_tmp, Aout_array, new_batchCount); //(int lda, int n, tipo **Ain, tipo **Aout, int new_batchCount)
      
      // Copio x in ptr_tmp
      copy_doubleptr_kernel<<< blocks3, threads1d >>>(x_array, ptr_tmp, batchCount);

      // Uso ptr_tmp come input per permutare x 
      permuta_idx<<< blocks3, threads1d >>>(ptr_tmp, x_array, new_batchCount); //(int lda, int n, tipo **Ain, tipo **Aout, int new_batchCount)
      
      // Copio bout in ptr_tmp
      copy_doubleptr_kernel<<< blocks3, threads1d >>>(bout_array, ptr_tmp, batchCount);
      
      // Uso ptr_tmp come input per permutare bout 
      permuta_idx<<< blocks3, threads1d >>>(ptr_tmp, bout_array, new_batchCount);

      //step 5: Ricorsione      
      rec_parasof(m, n,
		  Aout_array, ldaout,
		  Ain_array,  ldain,
		  T_array,    ldt,
		  Tau_array,
		  bout_array,
		  bin_array,
		  x_array,
		  ptr_tmp,
		  new_batchCount, depth+1, info);

    rec_parasof(m, n,
		Aout_array + new_batchCount, ldaout,
		Ain_array  + new_batchCount, ldain,
		T_array    + new_batchCount, ldt,
		Tau_array  + new_batchCount,
		bout_array + new_batchCount,
		bin_array  + new_batchCount,
		x_array + new_batchCount,
		ptr_tmp + new_batchCount,
		new_batchCount, depth+1, info);

    }
  }
  
  return;
}

__global__ void stampa_matrice(int m, int n, tipo ** MAT, int ldm, int slice_size, int it, tipo batchcount){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.z*slice_size + it;
  if ((tx>m) || (ty>n) || (bx > batchcount))
    return;

  tipo *dA = MAT[bx]; 

  if ((tx == 0 )&& (ty == 0) )
    printf("batch = %d \n %f %f \n %f %f \n %f %f \n %f %f \n", bx, dA[0*ldm + 0], dA[1*ldm + 0], dA[0*ldm + 1], dA[1*ldm + 1], dA[0*ldm + 2], dA[1*ldm + 2], dA[0*ldm + 3], dA[1*ldm + 3] );
    
  return;
}

__global__ void stampa_vettore(int n, tipo ** v, int slice_size, int it, tipo batchcount){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.z*slice_size + it;
  if ((tx>n) || (ty>0) || (bx > batchcount))
    return;

  tipo *dv = v[bx]; 

  if ((tx == 0 )&& (ty == 0) )
    printf("batch = %d \n %f %f \n", bx, dv[0], dv[1] );
    
  return;
}




void iter_parasof(int m, int n,
		  tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout,
		  tipo **T_array, int ldt, tipo **Tau_array,
		  tipo **bin_array, tipo **bout_array,
		  tipo **x_array,
		  tipo_int batchCount, int *info){
  /* versione iterativa di PARASOF con chiamate ai kernel gestite dalla CPU
   */
  if ((batchCount % 2) != 0){
    *info = -1;
    return;
  }
  
  int  it = 0;

  //ntcol_gemm    = 2;//magma_get_gemm_batched_ntcol( n );
  ntcol_geqr2   = magma_get_geqr2_batched_ntcol( m );
  // nblocks_gemm  = int(ceil(batchCount*1.0/ntcol_gemm));
  nblocks_geqr2 = int(ceil(batchCount*1.0/ntcol_geqr2));

  // printf("ntcol_geqr2 = %d, nblocks_geqr2 = %d \n \n", ntcol_geqr2, nblocks_geqr2);
  
  // blocks_gemm.z  = nblocks_gemm;
  blocks_geqr2.z = nblocks_geqr2;
  
  // threads_gemm.x = n, threads_gemm.y = n;
  threads_geqr2.x = WARP_SIZE;
  threads_geqr2.z = ntcol_geqr2;
  // threads_rec.x = m, threads_rec.y = n;

  
  dim3 blocks(1, 1, batchCount);
  dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2d(n, n, 1);
  dim3 threads2d_rec (m, n, 1);
  tipo_int offset = 1; 
  size_t shmem_gemm  = ( SLDA(n)*n*3 ) * sizeof(tipo), shmem_gemv  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
  tipo **tmp;

  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
   
  //printf("batchCount/2 = %d \n", batchCount/2);
  for(offset = 1; offset < batchCount/2; offset*=2){//batchCount/2 

    // Copio la matrice in Out
    copy_batched_kernel<<<blocks, threads2d_rec>>>(m, n, Ain_array, ldain, Aout_array, ldaout, batchCount);
    copy_batched_kernel<<<blocks, threads1d    >>>(n, 1, bin_array, int(1), bout_array, int(1), batchCount);

    // step 1. Batched compact WY representation of QR
    compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 0, 0, batchCount, offset);
    cudaDeviceSynchronize();

    // cudaDeviceSynchronize();
    // stampa_matrice<<<blocks, threads1d>>>(m, n, Ain_array, ldain, 1,0, batchCount);
    // cudaDeviceSynchronize();
    // stampa_vettore<<<blocks, threads1d>>>(n, bin_array, 1,0, batchCount); 
    // cudaDeviceSynchronize();
    
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    parasof_larfb_gemm_batched_smallsq_kernel<<<blocks, threads2d , shmem_gemm, s1 >>>(m, n,
										      Ain_array, ldain, 
										      T_array, ldt, 
										      Aout_array, ldaout,
										      batchCount, offset);

    // step 3. applicazione di Q = (I-VTV^T) a blocchi al rhs
    parasof_larfb_gemv_batched_smallsq_kernel<<<blocks, threads2d , shmem_gemv, s2 >>>(m, n,
											Ain_array, ldain, 
											T_array, ldt, 
											bin_array, bout_array,
											batchCount, offset);
    cudaDeviceSynchronize();
    tmp        = Ain_array;
    Ain_array  = Aout_array;
    Aout_array = tmp;
    
    tmp        = bin_array;
    bin_array  = bout_array;
    bout_array = tmp;

    it += 1;
  }

  
  cudaStreamDestroy(s1); 
  cudaStreamDestroy(s2);
     
  geqr2_batched_solver(m, n,
		       Ain_array, ldain,
		       Aout_array, ldaout,
		       T_array, ldt, Tau_array,
		       bin_array, x_array,
		       batchCount, offset);

  tmp        = Ain_array;
  Ain_array  = Aout_array;
  Aout_array = tmp;
  
  return;
}



void iter_parasof_FACT(int m, int n,
		       tipo **Ain_array, int ldain, tipo **Aout_array, int ldaout,
		       tipo **T_array, int ldt, tipo **Tau_array,
		       tipo **csrValM, tipo **csrColIndM,
		       tipo **bin_array, tipo **bout_array,
		       tipo **x_array, 
		       tipo_int batchCount, int *info){
  /* versione iterativa di PARASOF con chiamate ai kernel gestite dalla CPU
     NB: vengono calcolare le matrici M necessarie per la fattorizzazione della proposione
     
     Tutti i puntatori si riferiscono alla memoria GPU, tranne csrValM che punta a memoria CPU
   */
  if ((batchCount % 2) != 0){
    *info = -1;
    return;
  }
  
  int  it = 0;

  ntcol_geqr2   = magma_get_geqr2_batched_ntcol( m );
  nblocks_geqr2 = int(ceil(batchCount*1.0/ntcol_geqr2));

  blocks_geqr2.z = nblocks_geqr2;

  threads_geqr2.x = WARP_SIZE;
  threads_geqr2.z = ntcol_geqr2;

  dim3 blocks(1, 1, batchCount);
  dim3 threads1d(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2d(n, n, 1);
  dim3 threads2d_rec (m, n, 1);
  tipo_int offset = 1; 
  size_t shmem_gemm  = ( SLDA(n)*n*3 ) * sizeof(tipo), shmem_gemv  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
  tipo **tmp;

  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
   
  //printf("batchCount/2 = %d \n", batchCount/2);
  for(offset = 1; offset < batchCount/2; offset*=2){//batchCount/2 
    
    // Copio la matrice in Out
    copy_batched_kernel<<<blocks, threads2d_rec>>>(m, n, Ain_array, ldain, Aout_array, ldaout, batchCount);
    copy_batched_kernel<<<blocks, threads1d    >>>(n, 1, bin_array, int(1), bout_array, int(1), batchCount);

    // step 1. Batched compact WY representation of QR
    compactWY_batched(m, n, Ain_array, ldain, 0, 0, Aout_array, ldaout, 0, 0, T_array, ldt, Tau_array, 0, 0, batchCount, offset);
    cudaDeviceSynchronize();
    
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    // parasof_larfb_gemm_batched_smallsq_kernel<<<blocks, threads2d , shmem_gemm, s1 >>>(m, n,
    // 										       Ain_array, ldain, 
    // 										       T_array, ldt, 
    // 										       Aout_array, ldaout,
    // 										       batchCount, offset);
    parasof_larfb_calcolaM_gemm_batched_smallsq_kernel<<<blocks, threads2d , shmem_gemm, s1 >>>(m, n,
    												Ain_array, ldain, 
    												T_array, ldt, 
    												Aout_array, ldaout,
    												//d_csrValM, d_csrColIndM, 
												csrValM, csrColIndM, it,
    												batchCount, offset);

    
    // step 3. applicazione di Q = (I-VTV^T) a blocchi al rhs
    parasof_larfb_gemv_batched_smallsq_kernel<<<blocks, threads2d , shmem_gemv, s2 >>>(m, n,
										       Ain_array, ldain, 
										       T_array, ldt, 
										       bin_array, bout_array,
										       batchCount, offset);

    // qui un terzo stream si dovrebbe occupare del calcolo di M su GPU e di spedirla alla CPU
    
    cudaDeviceSynchronize();

    // cudaMemcpy(h_csrValM, d_csrValM, batchCount*m*n * sizeof(tipo), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_csrColIndM, d_csrColIndM, batchCount*m*n * sizeof(int), cudaMemcpyDeviceToHost);   
    // for (int ii = 0; ii< batchCount*m*n; ii++)
    //   printf("%f %d \n", h_csrValM[ii], h_csrColIndM[ii]);

    
    tmp        = Ain_array;
    Ain_array  = Aout_array;
    Aout_array = tmp;
    
    tmp        = bin_array;
    bin_array  = bout_array;
    bout_array = tmp;

    it += 1;
  }

  // free(h_csrValM); free(h_csrColIndM); cudaFree(d_csrValM); cudaFree(d_csrColIndM);
  
  cudaStreamDestroy(s1); 
  cudaStreamDestroy(s2);
     
  geqr2_batched_solver_FACT(m, n, Ain_array, ldain, Aout_array, ldaout, T_array, ldt,
			    Tau_array, bin_array, csrValM, csrColIndM, it, x_array, batchCount, offset);

  return;
}


void sof_forward(int m, int n,
		 tipo **Ain_array, int ldain, //array che contiene il sistema in ingresso e che in uscita conterra' i blocchi fill-in
		 tipo **Aout_array, int ldaout, //array che conterra' il sistema ridotto
		 tipo **T_array, int ldt,
		 tipo **R_array, int ldr, //array che conterra' i sistemi triangolari superiori
		 tipo **Tau_array,
		 tipo **bin_array, //rhs originale
		 tipo **bout_array, //rhs ridotto
		 int nb_slices,
		 int slice_size,
		 tipo_int batchCount,
		 int *info){
  /* Fase di forward-reduction della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa).
   */
  
  int it = 0;
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2 (n, n, 1);
  dim3 threads2_rec (m, n, 1);
  size_t shmem_gemm  = ( SLDA(n)*n*3 ) * sizeof(tipo), shmem_gemv  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
  
  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  
  for(it = 1; it < slice_size; it ++ ){
   
    // step 1. Batched compact WY representation of QR
    sof_compactWY_batched(m, n,
			  Ain_array, ldain, 0, 0, // Matrici di cui fare la QR, Q = I - VTV^T
			  Aout_array, ldaout, 0, 0, // Matrici V
			  R_array, ldr, 0, 0, // Matrici R
			  T_array, ldt, Tau_array, // matrici T
			  nb_slices, slice_size, it, batchCount);
    cudaDeviceSynchronize();
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    sof_larfb_gemm_batched_smallsq_kernel<<<blocks, threads2 , shmem_gemm, s1 >>>(m, n,
										  Aout_array, ldaout, // matrice V
										  T_array, ldt, 
										  Ain_array, ldain, // matrice fill-in
										  nb_slices, slice_size, it, batchCount);
    sof_larfb_gemv_batched_smallsq_kernel<<<blocks, threads2 , shmem_gemv, s2 >>>(m, n,
										  Aout_array, ldaout, // matrice V
										  T_array, ldt, 
										  bin_array, // B aggiornato
										  nb_slices, slice_size, it, batchCount);
    //QUESTA BARRIERA E' IMPORTANTISSIMA, NON CANCELLARE
    cudaDeviceSynchronize();
  }
  
  // copiare i blocchi opportuni in Aout
  // copiare le parti giuste del RHS
  dim3 blocks2(1, 1, int(nb_slices+1));
  sof_assembly_rd_babd_batched_kernel<<<blocks2, threads2_rec>>>(m, n, Ain_array, ldain,
								 Aout_array, ldaout,
								 bin_array, bout_array,
								 nb_slices, slice_size, batchCount);

  // cudaDeviceSynchronize();
  // printf("Sistema ridotto: \n");
  // stampa_matrice<<<blocks2, threads>>>(m, n, Aout_array, ldaout, 1,0, nb_slices+1);
  // cudaDeviceSynchronize();
  // stampa_vettore<<<blocks2, threads>>>(n, bout_array, 1,0, nb_slices+1); 
  
  cudaDeviceSynchronize();
  
  cudaStreamDestroy(s1); 
  cudaStreamDestroy(s2);

  return;
}

__global__ void sof_scambiablocchi_FACT(int n,
					tipo **A_array, int lda,
					int nb_slices,
					int slice_size, tipo_int batchCount ){ 
  // scambia il blocco in basso del primo batch e il blocco in basso bell'ultimo batch di una stessa slice
  int sliceId = blockIdx.z * blockDim.z; 
  int batchId_first = sliceId * slice_size;
  int batchId_last = (sliceId+1) * slice_size - 1;
  int tx = threadIdx.x, ty = threadIdx.y;

  
  if ((tx>= n) || (ty >=n) || ( batchId_first > batchCount) || ( batchId_last > batchCount))
    return;

  // if ((tx == 0) && (ty == 0))
  //   printf("sliceId %d batchId_first %d batchId_last %d \n ", sliceId, batchId_first, batchId_last);
    
  tipo *dA_first = A_array[batchId_first] + n; //  dA += aj * ldda + ai
  tipo *dA_last  = A_array[batchId_last]  + n;  
  tipo tmp = dA_first[ty * lda + tx];
  dA_first[ty * lda + tx] = dA_last[ty * lda + tx];
  dA_last[ty * lda + tx]  = tmp;
  
}

void sof_forward_FACT(int m, int n,
		      tipo **Ain_array, int ldain, //array che contiene il sistema in ingresso e che in uscita conterra' i blocchi fill-in e il sistema ridotto
		      tipo **Aout_array, int ldaout, //array che conterra' V 
		      tipo **T_array, int ldt,
		      tipo **R_array, int ldr, //array che conterra' i sistemi triangolari superiori
		      tipo **Tau_array,
		      tipo **bin_array, //rhs originale
		      int nb_slices,
		      int slice_size,
		      tipo_int batchCount){

  /* Fase di forward-reduction della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa).
   */
  
  int it = 0;
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  dim3 threads2 (n, n, 1);
  dim3 threads2_rec (m, n, 1);
  size_t shmem_gemm  = ( SLDA(n)*n*3 ) * sizeof(tipo), shmem_gemv  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 
  
  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  
  for(it = 1; it < slice_size; it ++ ){ 
    
    // step 1. Batched compact WY representation of QR
    sof_compactWY_batched_FACT(m, n,
  			       Ain_array, ldain, 0, 0, // Matrici di cui fare la QR, Q = I - VTV^T
  			       Aout_array, ldaout, 0, 0, // Matrici V
  			       R_array, ldr, 0, 0, // Matrici R
  			       T_array, ldt, Tau_array, // matrici T
  			       nb_slices, slice_size, it, batchCount);
    cudaDeviceSynchronize();
    // step 2. applicazione di Q = (I-VTV^T) a blocchi
    sof_larfb_gemm_batched_smallsq_kernel_FACT<<<blocks, threads2 , shmem_gemm, s1 >>>(m, n,
    										       Aout_array, ldaout, // matrice V
    										       T_array, ldt, 
    										       Ain_array, ldain, // matrice fill-in
    										       nb_slices, slice_size, it, batchCount);
    sof_larfb_gemv_batched_smallsq_kernel_FACT<<<blocks, threads2 , shmem_gemv, s2 >>>(m, n,
    										       Aout_array, ldaout, // matrice V
    										       T_array, ldt, 
    										       bin_array, // B aggiornato
    										       nb_slices, slice_size, it, batchCount);
    
    //QUESTA BARRIERA E' IMPORTANTISSIMA, NON CANCELLARE
    cudaDeviceSynchronize();
  }

  // SCAMBIARE I BLOCCHI DI AIN
  sof_scambiablocchi_FACT<<<blocks, threads2  >>>(n,
						  Ain_array, ldain,
						  nb_slices,
						  slice_size, batchCount);
  // cudaDeviceSynchronize();
  // blocks.z = int(nb_slices+1);
  // printf("Sistema ridotto: \n");
  // stampa_matrice<<<blocks, threads>>>(m, n, Ain_array, ldain, slice_size, 0, batchCount);
  // cudaDeviceSynchronize();
  // stampa_vettore<<<blocks, threads>>>(n, bin_array,  slice_size, 0, batchCount); 

  cudaDeviceSynchronize();
  
  cudaStreamDestroy(s1); 
  cudaStreamDestroy(s2);

  return;
}

void sof_forward_aggiornaRHS(int m, int n,
			     tipo **Aout_array, int ldaout, //array che conterra' V 
			     tipo **T_array, int ldt,
			     tipo **bin_array, //rhs originale
			     int nb_slices,
			     int slice_size,
			     tipo_int batchCount){

  /* Fase di forward-reduction della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa).
   */
  
  int it = 0;
  dim3 blocks(1, 1, nb_slices);
  dim3 threads2 (n, n, 1);
  size_t shmem_gemv  = ( SLDA(n)*n + 2*n ) * sizeof(tipo); 

  for(it = 1; it < slice_size; it ++ ){ 
    sof_larfb_gemv_batched_smallsq_kernel_FACT<<<blocks, threads2 , shmem_gemv >>>(m, n,
										   Aout_array, ldaout, // matrice V
										   T_array, ldt, 
										   bin_array, // B aggiornato
										   nb_slices, slice_size, it, batchCount);
    
    //QUESTA BARRIERA E' IMPORTANTISSIMA, NON CANCELLARE
    cudaDeviceSynchronize();
  }

  return;
}

void sof_backward(int m, int n,
		  tipo **Ain_array, int ldain, //array che contiene i blocchi fill-in
		  tipo **R_array, int ldr, //array che contiene i sistemi triangolari superiori
		  tipo **bin_array, //rhs originale
		  tipo **x_array,
		  int nb_slices,
		  int slice_size,
		  tipo_int batchCount){

   /* Fase di back-substitution della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa) 
      che calcola le variabili eliminate nella prima fase di riduzione.
   */
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  size_t shmem  = ( 2*n ) * sizeof(tipo);
  sof_dtrsv_batched_kernel<<< blocks, threads, shmem >>>(n,
   							 Ain_array, ldain,
    							 R_array, ldr,
   							 bin_array, x_array,
   							 nb_slices, slice_size, batchCount);

  
  return;
}

void sof_backward_SOLV(int m, int n,
		       tipo **Ain_array, int ldain, //array che contiene i blocchi fill-in
		       tipo **R_array, int ldr, //array che contiene i sistemi triangolari superiori
		       tipo **bin_array, //rhs originale
		       tipo **x_array,
		       int nb_slices,
		       int slice_size,
		       tipo_int batchCount){
  
  /* Fase di back-substitution della versione finale di PARASOF (implementazione ibrida ricorsiva/iterativa) 
     che calcola le variabili eliminate nella prima fase di riduzione.
  */
  dim3 blocks(1, 1, nb_slices);
  dim3 threads(WARP_SIZE, 1, 1); //threads(BLOCK_SIZE), BLOCK_SIZE = il piu' piccolo multiplo di 32 >= max(m,n)
  size_t shmem  = ( 2*n ) * sizeof(tipo);
  sof_dtrsv_batched_kernel_SOLV<<< blocks, threads, shmem >>>(n,
							      Ain_array, ldain,
							      R_array, ldr,
							      bin_array, x_array,
							      nb_slices, slice_size, batchCount);

  
  return;
}


extern "C" void parasof_hybrid(int m, int n,
			       tipo **Ain, tipo **Aout,
			       tipo  **d_TArray, tipo  **d_RArray, tipo **d_TauArray,
			       tipo **bin, tipo **bout,
			       tipo **xin, int nb_slices, tipo_int batchCount,
			       int *d_info, tipo *d_elapsed){
   /* Versione finale di PARASOF: implementazione ibrida ricorsiva/iterativa. 
     Il sistema iniziale viene ridotto ad una dimensione tale per cui l'implementazione iterativa puo' essere chiamata con la garanzia che le 
     QR batched siano eseguite simultaneamente. Segue una fase di back-substitution per calcolare le variabili eliminate nella prima fase di riduzione.
   */
  int info[1];
  cudaEvent_t e_start, e_stop;
  tipo **h_Aptr = 0;
  tipo **Aptr = NULL;
  tipo **h_bptr = 0;
  tipo **bptr = NULL;
 
  float elapsed;
  tipo **xout;
   
  if ((int(batchCount-1) % (nb_slices)) != 0){
    *info = -1;
    printf("ERRORE: batchCount-1 = %d non e' divisibile per nb_slices = %d \n", int(batchCount-1), nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }

  int slice_size = (int(batchCount-1) / (nb_slices));
  if (slice_size < 2){
    *info = -1;
    printf("ERRORE: nb_slices = %d e' piu' piccolo di 2 \n", nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }
  int batchCount_new = nb_slices+1;
  
  dim3 blocks((int) (batchCount_new/WARP_SIZE +1), 1, 1);
  dim3 threads(WARP_SIZE, 1, 1);

  h_Aptr = (tipo**)malloc(batchCount_new*sizeof(*h_Aptr));
  h_bptr = (tipo**)malloc(batchCount_new*sizeof(*h_bptr));
 
  for (int i = 0; i < batchCount_new; i++){
    cudaMalloc((void**)&h_Aptr[i], (m*n)*sizeof(tipo));
    cudaMalloc((void**)&h_bptr[i], n*sizeof(tipo));
  }

  cudaMalloc((void**)&xout, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&Aptr, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&bptr, batchCount_new*sizeof(tipo*));
  
  cudaMemcpy(bptr, h_bptr, batchCount_new * sizeof(tipo*), cudaMemcpyHostToDevice);
  cudaMemcpy(Aptr, h_Aptr, batchCount_new * sizeof(tipo*), cudaMemcpyHostToDevice);

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  
  //  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);

  cudaProfilerStart();

  sof_forward(m, n, //int m, int n,
	      Ain, m, //  tipo **Ain_array, int ldain,  // array che contiene il sistema in ingresso e che in uscita conterra' i blocchi fill-in
	      Aout, m, //  tipo **Aout_array, int ldaout,  //array che conterra' il sistema ridotto
	      d_TArray, n, //  tipo **T_array, int ldt,
	      d_RArray, n, //  tipo **R_array, int ldr, //array che conterra' i sistemi triangolari superiori
	      d_TauArray, // tipo **Tau_array,
	      bin, // tipo **bin_array //rhs originale
	      bout, //  tipo **bout_array //rhs ridotto
	      nb_slices,
	      slice_size,
	      batchCount,
	      d_info);
  
  sof_selectXarrays_doubleptr_kernel<<<blocks, threads>>>(xin, xout, nb_slices, slice_size, batchCount);
  
  iter_parasof(m, n, Aout, m, Aptr, m, d_TArray, n, d_TauArray, bout, bptr, xout, nb_slices+1, d_info);
  
  sof_backward(m, n, Ain, m,
	       d_RArray, n, 
	       bin,
	       xin,
	       nb_slices,
	       slice_size,
	       batchCount);
  
  //cudaProfilerStop();
 
  cudaDeviceSynchronize();

  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);  
  clock_t stop = clock();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;

  //tipo telapsed = elapsed;
  cudaMemcpy(d_elapsed, &time, sizeof(tipo), cudaMemcpyHostToDevice);
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);

  for(int i=0; i < batchCount_new; i++) {
    cudaFree(h_Aptr[i]);
    cudaFree(h_bptr[i]);
   }
  cudaFree(Aptr); 
  cudaFree(bptr); 
  cudaFree(xout);
  free(h_Aptr);
  free(h_bptr);
  
  return;
}

extern "C" void parasof_hybrid_FACT(int m, int n,
				    tipo **Ain,
				    tipo **Aout, //stessa dim di Ain
				    tipo **T, // tanti batch quantidi Ain
				    tipo **R, tipo **Tau,
				    tipo **csrValM, tipo **csrColIndM,
				    tipo **bin, 
				    tipo **x, int nb_slices, tipo_int batchCount,
				    int *d_info, tipo *d_elapsed){
  /* Versione finale di PARASOF: implementazione ibrida ricorsiva/iterativa. 
     Il sistema iniziale viene ridotto ad una dimensione tale per cui l'implementazione iterativa puo' essere chiamata con la garanzia che le 
     QR batched siano eseguite simultaneamente. Segue una fase di back-substitution per calcolare le variabili eliminate nella prima fase di riduzione.
  */
  int info[1];
  cudaEvent_t e_start, e_stop;
  
  float elapsed;
  tipo **Ain_red;
  tipo **Aout_red;
  tipo **T_red;
  tipo **b_red;
  tipo **x_red;
  tipo **bout = NULL;
  tipo **h_bout;
   
  if ((int(batchCount-1) % (nb_slices)) != 0){
    *info = -1;
    printf("ERRORE: batchCount-1 = %d non e' divisibile per nb_slices = %d \n", int(batchCount-1), nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }
  
  int slice_size = (int(batchCount-1) / (nb_slices));
  if (slice_size < 2){
    *info = -1;
    printf("ERRORE: nb_slices = %d e' piu' piccolo di 2 \n", nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }
  int batchCount_new = nb_slices+1;
  
  dim3 blocks((int) (batchCount_new/WARP_SIZE +1), 1, 1);
  dim3 threads(WARP_SIZE, 1, 1);

  h_bout = (tipo**)malloc(batchCount_new*sizeof(*h_bout));
 
  for (int i = 0; i < batchCount_new; i++){
    cudaMalloc((void**)&h_bout[i], n*sizeof(tipo));
  }
  cudaMalloc((void**)&bout, batchCount_new*sizeof(tipo*));
  cudaMemcpy(bout, h_bout, batchCount_new * sizeof(tipo*), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&x_red, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&b_red, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&T_red,  batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&Ain_red,  batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&Aout_red, batchCount_new*sizeof(tipo*));
  

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  
  //  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);

  cudaProfilerStart();

  sof_forward_FACT(m, n, //int m, int n,
		   Ain, m, //  tipo **Ain_array, int ldain,  // array che contiene il sistema in ingresso e che in uscita conterra' i blocchi fill-in
		   Aout, m, //  tipo **Aout_array, int ldaout,  //array che conterra' il sistema ridotto e le V
		   T, n, //  tipo **T_array, int ldt,
		   R, n, //  tipo **R_array, int ldr, //array che conterra' i sistemi triangolari superiori
		   Tau, // tipo **Tau_array,
		   bin, // tipo **bin_array //rhs originale
		   nb_slices,
		   slice_size,
		   batchCount);

  sof_selectXarrays_doubleptr_kernel_FACT<<<blocks, threads>>>(Ain, Ain_red, Aout, Aout_red, T, T_red,
							       bin, b_red, x, x_red,
							       nb_slices, slice_size, batchCount);

  // dim3 blocks2( 1, 1, batchCount_new);
  // cudaDeviceSynchronize();
  // printf("Ain_red: \n");
  // stampa_matrice<<<blocks2, threads>>>(m, n, Ain_red, m, 1,0, batchCount_new);
  // cudaDeviceSynchronize();
  // printf("Aout_red: \n");
  // stampa_matrice<<<blocks2, threads>>>(m, n, Aout_red, m, 1,0, batchCount_new);
  // cudaDeviceSynchronize();

  iter_parasof_FACT(m, n,
  		    Ain_red, m, Aout_red, m, 
  		    T_red, n, Tau,
  		    csrValM, csrColIndM,
  		    b_red, bout,
  		    x_red, nb_slices+1, d_info);
  
  sof_backward_SOLV(m, n, Ain, m,
  		    R, n, 
  		    bin,
  		    x, //o x
  		    nb_slices,
  		    slice_size,
  		    batchCount);
  
  cudaProfilerStop();
  cudaDeviceSynchronize();
  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);
  clock_t stop = clock();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(d_elapsed, &time, sizeof(tipo), cudaMemcpyHostToDevice);
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);


  cudaFree(Ain_red);
  cudaFree(Aout_red);
  cudaFree(T_red);
  cudaFree(b_red); 
  cudaFree(x_red);

  for(int i=0; i < batchCount_new; i++) {
    cudaFree(h_bout[i]);
   }
  cudaFree(bout); 
  free(h_bout);
  
  return;
}



extern "C" void parasof_new_FACT( int m, int n,
				  tipo **Ain, tipo **Aout,			     
				  tipo  **d_TArray_dev, tipo **d_TauArray_dev, //tipo **ptr_tmp,
				  tipo **csrValM, tipo **csrColIndM,
				  tipo **bin, tipo **bout,
				  tipo **x, 
				  tipo_int batchCount, int *d_info){
  /* Seconda versione di PARASOF: implementazione iterativa. 
     Non molto efficiente, poiche' il parallelismo dell'applicazione e' limitato dal numero massimo di QR che la GPU puo' fisicamente eseguire simultaneamente.
   */
  
  int info[1];
  cudaEvent_t e_start, e_stop;
  float elapsed;

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  int driver_version = 0, runtime_version = 0;

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  
  printf("Driver Version: %d\n"
	 "Runtime Version: %d\n",
	 driver_version, runtime_version);
  
  //  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);

  cudaProfilerStart();

  iter_parasof_FACT(m, n, Ain, m, Aout, m,
		    d_TArray_dev, n, d_TauArray_dev,
		    csrValM, csrColIndM,
		    bin, bout,
		    x, batchCount, d_info);

  cudaProfilerStop();
  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);
  cudaDeviceSynchronize();
  clock_t stop = clock();
  //cudaProfilerStop();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;
  
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);


  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  
  return;
}

extern "C" void parasof_new( int m, int n,
			     tipo **Ain, tipo **Aout,			     
			     tipo  **d_TArray_dev, tipo **d_TauArray_dev, //tipo **ptr_tmp,
			     tipo **bin, tipo **bout,
			     tipo **x,
			     tipo_int batchCount, int *d_info){

  /* Seconda versione di PARASOF: implementazione iterativa. 
     Non molto efficiente, poiche' il parallelismo dell'applicazione e' limitato dal numero massimo di QR che la GPU puo' fisicamente eseguire simultaneamente.
  */


  int info[1];
  cudaEvent_t e_start, e_stop;
  float elapsed;

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  
  //  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);

  cudaProfilerStart();
  
  iter_parasof(m, n,
	       Ain, m, Aout, m,
	       d_TArray_dev, n, d_TauArray_dev,
	       bin, bout, x,
	       batchCount, info);

  cudaProfilerStop();
  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);
  cudaDeviceSynchronize();
  clock_t stop = clock();
  //cudaProfilerStop();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;

  //cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_info, info, sizeof(int), cudaMemcpyHostToDevice);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);


  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  
  return;
}


// CSR-SpMV kernel
__global__ void 
babd_gecsrmv_kernel(int num_rows, 
		    int num_cols,
		    int n,
		    tipo alpha, 
		    tipo** dvalptr, 
		    tipo** dcolindptr,
		    int it,
		    tipo * dx,
		    tipo beta, 
		    tipo * dy)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;
    
    tipo* dval = dvalptr[it]; 
    tipo* dcolind = dcolindptr[it]; 

    // if (row == 0){
    //   printf("it %d \n", it);
    //   printf("Prima riga %f %f %f %f, indici col %d %d %d %d \n ", dval[0], dval[1], dval[2], dval[3], int(dcolind[0]), int(dcolind[1]), int(dcolind[2]), int(dcolind[3]));
    //   printf("Seconda riga %f %f %f %f, indici col %d %d %d %d \n ", dval[4], dval[5], dval[6], dval[7], int(dcolind[4]), int(dcolind[5]), int(dcolind[6]), int(dcolind[7]));
    //   printf("Terza riga %f %f %f %f, indici col %d %d %d %d \n ", dval[8], dval[9], dval[10], dval[11], int(dcolind[8]), int(dcolind[9]), int(dcolind[10]), int(dcolind[11]));
    //   for (int ii=0; ii<num_cols; ii++)
    // 	printf("x[%d] = %f \t", ii, dx[ii]);
    //   printf("\n");
    // }  
    
    if(row<num_rows){
      tipo dot = d_zero;
      int start = (2*n)*row ;
      int end   = start+2*n;
      for( j=start; j<end; j++)
	dot += dval[ j ] * dx[ int(dcolind[j]) ];
      // printf("row %d; start %d end %d dot %f \n", row, start, end, dot);
      dy[ row ] =  dot *alpha + beta * dy[ row ];
    }
}

__global__ void copyvec_batched2full(int n, int batchsize, tipo **b_array, tipo *b){
  /*kernel che esegue la copia di un array batched (= zone di memoria indicate dal doppio puntatore) dove gli array puntati sono matrici quadrate. 
    Utilizzato nella risoluzione dei sistemi lineari 2x2 */
  
  int tx = threadIdx.x;
  int idx = batchsize*blockIdx.x+tx;

  //printf("tx %d blkidx %d  \n", tx, blockIdx.x);
  
  if ((tx >= batchsize) || (idx >= n))
    return;
  
  b[idx] = b_array[blockIdx.x][tx];
  
  //printf("tx %d blkidx %d :b[idx] %f, b_array[blockIdx.x][tx] %f \n", tx, blockIdx.x, b[idx], b_array[blockIdx.x][tx]);
  
  return;
}

__global__ void copyvec_full2batched(int n, int batchsize, tipo **b_array, tipo *b){
  /*kernel che esegue la copia di un array batched (= zone di memoria indicate dal doppio puntatore) dove gli array puntati sono matrici quadrate. 
    Utilizzato nella risoluzione dei sistemi lineari 2x2 */
  
  int tx = threadIdx.x;
  int idx = batchsize*blockIdx.x+tx;
  
  if ((tx >= batchsize) || (idx >= n))
    return;
  
  b_array[blockIdx.x][tx] = b[idx];
  //printf("tx %d blkidx %d : %f \n", tx, blockIdx.x, b[idx]);
  
  return;
}

extern "C" void iter_parasof_SOLV(int m, int n,
				 tipo **T_array, int ldt, // sistemi triangolari contenuti in Aout_array, cambio nome x sottolineare che sono triangolari
				 tipo **b_array,
				 tipo **csrValM, tipo **csrColIndM, 
				 tipo **x_array, tipo_int batchCount){
  

  int it, steps =  int(log(batchCount)/log(2)), N = batchCount*n;
  tipo *b, *x, *tmp;

  
  cudaMalloc((void**)&b, batchCount*n*sizeof(tipo));
  cudaMalloc((void**)&x, batchCount*n*sizeof(tipo));
  
  dim3 grid( int(N/BLOCK_SIZE)+1 );
  dim3 threads (BLOCK_SIZE);

  dim3 blocks(batchCount, 1, 1);
  dim3 threads1d(WARP_SIZE, 1, 1);

  int offset = int(batchCount/2);
  dim3 block(offset, 1, 1);
  dim3 threads2  (n, 1, 1);
  size_t shmem = sizeof(tipo)*n;

  // printf(" BatchCount = %d, steps = %d, N = %d, grid %d \n", batchCount, steps, N, int(N/BLOCK_SIZE)+1 );

  // copy b_array in b
  copyvec_batched2full<<< blocks, threads1d>>>(N, n, b_array, b);

  // tipo *h_b;
  // int *h_indptr;
  // h_b = (tipo*)malloc(N*sizeof(*h_b));
  // h_indptr = (int*)malloc((n*batchCount+1)*sizeof(*h_b));
  // cudaMemcpy(h_indptr, csrRowPtrM, (n*batchCount+1) * sizeof(int), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_b, b, N * sizeof(tipo), cudaMemcpyDeviceToHost);
  // printf("rhs = ");
  // for (int ii = 0; ii < N; ii++ )
  //   printf("%f \t", h_b[ii]);
  // printf("\n");
  // printf("indptr = " );
  // for (int ii = 0; ii < (n*batchCount+1); ii++ )
  //   printf("%d \t", h_indptr[ii]);
  // printf("\n");
  
  for (it = 0;  it < steps; it++){
    babd_gecsrmv_kernel<<< grid, threads, 0>>>(N, N, n, one, csrValM, csrColIndM, it, b, zero, x);
    // cudaMemcpy(h_b, x, N * sizeof(tipo), cudaMemcpyDeviceToHost);
    // printf("M*x = ");
    //   for (int ii = 0; ii < N; ii++ )
    // 	printf("%f \t", h_b[ii]);
    // printf("\n");
    tmp = b;
    b = x;
    x = tmp;
  }

  // copy b in x_array
  copyvec_full2batched<<< blocks, threads1d>>>(N, n, b_array, b);

  dtrsv_last_call_kernel<<<block, threads2, shmem>>>(n,  T_array, ldt, b_array, x_array, offset);

  cudaFree(x);
  cudaFree(b);

  
  return;
  
}


extern "C" void parasof_hybrid_SOLV(int m, int n,
				    tipo **Ain, // fill in e ristema ridotto
				    tipo **Aout, // matrici V
				    tipo **T, 
				    tipo **R, // tipo **Tau,
				    tipo **csrValM, tipo **csrColIndM,
				    tipo **b, 
				    tipo **x,
				    int nb_slices, tipo_int batchCount,
				    int *d_info, tipo *d_elapsed)
{
  /* Versione finale di PARASOF: implementazione ibrida ricorsiva/iterativa. 
     Il sistema iniziale viene ridotto ad una dimensione tale per cui l'implementazione iterativa puo' essere chiamata con la garanzia che le 
     QR batched siano eseguite simultaneamente. Segue una fase di back-substitution per calcolare le variabili eliminate nella prima fase di riduzione.
  */
  int info[1];
  cudaEvent_t e_start, e_stop;
  
  float elapsed;
  tipo **Ain_red;
  tipo **Aout_red;
  tipo **T_red;
  tipo **b_red;
  tipo **x_red;
   
  if ((int(batchCount-1) % (nb_slices)) != 0){
    *info = -1;
    printf("ERRORE: batchCount-1 = %d non e' divisibile per nb_slices = %d \n", int(batchCount-1), nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }
  
  int slice_size = (int(batchCount-1) / (nb_slices));
  if (slice_size < 2){
    *info = -1;
    printf("ERRORE: nb_slices = %d e' piu' piccolo di 2 \n", nb_slices);
    cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
    return;
  }
  int batchCount_new = nb_slices+1;
  
  dim3 blocks((int) (batchCount_new/WARP_SIZE +1), 1, 1);
  dim3 threads(WARP_SIZE, 1, 1);
  

  cudaMalloc((void**)&x_red, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&b_red, batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&T_red,  batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&Ain_red,  batchCount_new*sizeof(tipo*));
  cudaMalloc((void**)&Aout_red, batchCount_new*sizeof(tipo*));
  

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  
  //  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);

  cudaProfilerStart();

  // aggiorna RHS
  sof_forward_aggiornaRHS(m, n, //int m, int n,
			  Aout, m, //  tipo **Aout_array, int ldaout,  //array che conterra' il sistema ridotto e le V
			  T, n,
			  b, // tipo **bin_array //rhs originale
			  nb_slices,
			  slice_size,
			  batchCount);

  sof_selectXarrays_doubleptr_kernel_FACT<<<blocks, threads>>>(Ain, Ain_red, Aout, Aout_red, T, T_red,
							       b, b_red, x, x_red,
							       nb_slices, slice_size, batchCount);


  iter_parasof_SOLV(m, n,
		    Ain_red, m, // sistemi triangolare contenuti in Aout_array, cambio nome x sottolineare che sono triangolari
		    b_red,
		    csrValM, csrColIndM,
		    x_red, nb_slices+1);
  
  sof_backward_SOLV(m, n, Ain, m, //fill-in
  		    R, n, 
  		    b,
  		    x, 
  		    nb_slices,
  		    slice_size,
  		    batchCount);
  
  cudaProfilerStop();
 
  cudaDeviceSynchronize();
  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);  
  clock_t stop = clock();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(d_elapsed, &time, sizeof(tipo), cudaMemcpyHostToDevice);
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);


  cudaFree(Ain_red);
  cudaFree(Aout_red);
  cudaFree(T_red);
  cudaFree(b_red); 
  cudaFree(x_red);

  
  return;
}

extern "C" void parasof( int m, int n, tipo **Ain, tipo **Aout, tipo **bin, tipo **bout, tipo **x, tipo_int batchCount){
  /* Prima versione di PARASOF: implementazione ricorsiva che utilizza il Dynamic Parallelism. 
     Non molto efficiente, in quanto il DP non garantisce che il flusso di programma venga eseguito come vorrebbe l'utente. 
     Inoltre le GPU supportano un numero massimo di chiamate concorrenti a kernel che limita il parallelismo dell'applicazione.
   */
  tipo **d_TauArray = 0;
  tipo **d_TauArray_dev = NULL;
  tipo **d_TArray = 0;
  tipo **d_TArray_dev = NULL;
  tipo **ptr_tmp  = NULL;
  int *d_info, info[1];
  cudaEvent_t e_start, e_stop;
  float elapsed;

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  
  *info = 0;
  cudaMalloc((void**)&d_info, sizeof(int));
  cudaMemcpy(d_info, info, sizeof(int), cudaMemcpyHostToDevice);

  d_TauArray = (tipo**)malloc(batchCount*sizeof(*d_TauArray));
  d_TArray = (tipo**)malloc(batchCount*sizeof(*d_TArray));
  // d_bArray = (tipo**)malloc(batchCount*sizeof(*d_bArray));
 
  for (int i = 0; i < batchCount; i++){
    cudaMalloc((void**)&d_TauArray[i],   n*sizeof(tipo));
    cudaMalloc((void**)&d_TArray[i], (n*n)*sizeof(tipo));
    // cudaMalloc((void**)&d_bArray[i],   n*sizeof(tipo));
  }

  cudaMalloc((void**)&d_TauArray_dev,  batchCount*sizeof(tipo*));
  cudaMalloc((void**)&d_TArray_dev,    batchCount*sizeof(tipo*));

  cudaMalloc((void**)&ptr_tmp,         batchCount*sizeof(tipo*));

  //  cudaMalloc((void**)&d_bArray_dev,    batchCount*sizeof(tipo*));
  cudaMemcpy(d_TauArray_dev, d_TauArray, batchCount * sizeof(tipo*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_TArray_dev, d_TArray, batchCount * sizeof(tipo*), cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_bArray_dev, d_bArray, batchCount * sizeof(tipo*), cudaMemcpyHostToDevice);

  cudaProfilerStart();
  clock_t start = clock();
  cudaEventRecord(e_start, 0);
  cdp_parasof<<<1,1>>>(m, n, Ain, m, Aout, m, d_TArray_dev, n, d_TauArray_dev, bin, bout, x, ptr_tmp, batchCount, 0, d_info);
  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);
  cudaDeviceSynchronize();
  clock_t stop = clock();
  cudaProfilerStop();
  cudaEventElapsedTime(&elapsed, e_start, e_stop);
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;
  
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);	
  //printf("Chiamata terminata, info = %d, time clock = %f s, time event = %.2f ms \n", *info, time, elapsed);


  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  
  for(int i=0; i<batchCount; i++) {
    cudaFree(d_TauArray[i]);
    cudaFree(d_TArray[i]);
    //   cudaFree(d_bArray[i]);
   }
  cudaFree(d_TauArray_dev);
  cudaFree(d_TArray_dev);
  cudaFree(ptr_tmp);
  cudaFree(d_info);
  // cudaFree(d_bArray_dev);
  
  free(d_TauArray);
  free(d_TArray);
  //  free(d_bArray);
  
  return;

}


