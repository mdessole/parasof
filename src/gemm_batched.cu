/* dgemm_batched.cu -- modified by MAGMA http://icl.cs.utk.edu/magma/
   M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
*/

__device__ void magma_max_reduce(int n, int i, tipo* x )
{
  __syncthreads();
  if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max( x[i], x[i+1024] ); }  __syncthreads(); }
  if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max( x[i], x[i+ 512] ); }  __syncthreads(); }
  if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max( x[i], x[i+ 256] ); }  __syncthreads(); }
  if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max( x[i], x[i+ 128] ); }  __syncthreads(); }
  if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max( x[i], x[i+  64] ); }  __syncthreads(); }
  if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max( x[i], x[i+  32] ); }  __syncthreads(); }
  // probably don't need __syncthreads for < 16 threads
  // because of implicit warp level synchronization.
  if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max( x[i], x[i+  16] ); }  __syncthreads(); }
  if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max( x[i], x[i+   8] ); }  __syncthreads(); }
  if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max( x[i], x[i+   4] ); }  __syncthreads(); }
  if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max( x[i], x[i+   2] ); }  __syncthreads(); }
  if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_reduce

__device__ void magma_sum_reduce(int n, int i, tipo* x )
{
  __syncthreads();
  if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
  if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
  if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
  if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
  if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
  if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
  // probably don't need __syncthreads for < 16 threads
  // because of implicit warp level synchronization.
  if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
  if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
  if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
  if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
  if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce

/******************************************************************************/
/*
  lapack slarfg, compute the norm, scale and generate the householder vector   
  assume swork, sscale, scale are already allocated in shared memory
  BLOCK_SIZE is set outside, the size of swork is BLOCK_SIZE

*/


__device__ void
device_gemm_batched_smallsq_kernel(int n, tipo alpha, tipo* sA, int slda, tipo* sB, int sldb, tipo beta, tipo* sC, int sldc){
  /* C = alpha A*B + beta*C */

  int tx = threadIdx.x, ty = threadIdx.y;

  if ((tx >= n) || (ty >= n))
    return;
  
  tipo rC; 
  if(beta != d_zero){
    rC = beta * sC[ty * sldc + tx];
  }else
    rC = d_zero;

  // multiply
  tipo rTmp = d_zero;
#pragma unroll
  for(int j = 0; j < n; j++){
    rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
  }
  rC += alpha * rTmp;

  // write from rC
  sC[ty * sldc + tx] = rC;
  
  return;
}

__global__ void
gemm_batched_smallsq_kernel(int transA, int transB,
			    int N,
			    tipo alpha,
			    tipo const * const * dA_array, int ai, int aj, int ldda, 
			    tipo const * const * dB_array, int bi, int bj, int lddb,
			    tipo beta,
			    tipo**  dC_array, int ci, int cj, int lddc, 
			    tipo_int batchCount)
{

  /* dC = alpha*op(dA)*op(dB) + beta(dC) */
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,batchCount -1
  
  // tipo alpha = d_one, beta;
  int batchid = bx * blockDim.z + tz; // = bx per ogni blocco
  //int batchid = bx; // = bx per ogni blocco

  if(batchid >= batchCount) return;
  
  const tipo* __restrict__ dA;
  const tipo* __restrict__ dB;
  tipo* __restrict__ dC;
  
  const int slda = SLDA(N); 
  const int sldb = SLDA(N);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + blockDim.z * slda * N);
  sA += tz * slda * N;
  sB += tz * sldb * N;
    
  dA = dA_array[batchid] + aj * ldda + ai; // dA = dA_array[batchid][ai:,aj:]
  dB = dB_array[batchid] + bj * lddb + bi; // dB = dB_array[batchid][bi:,bj:]
  dC = dC_array[batchid] + cj * lddc + ci;  // dC = dC_array[batchid][ci:,cj:]
    
  // read A & B 
  if(transA == 0){
    sA[ty * slda + tx] = dA[ty * ldda + tx];
  }
  else{
    sA[tx * slda + ty] = (transA == 1) ? dA[ty * ldda + tx] : dA[ty * ldda + tx] ;
  }
    
  if(transB == 0){
    sB[ty * sldb + tx] = dB[ty * lddb + tx];
  }
  else{
    sB[tx * sldb + ty] = (transB == 1) ? dB[ty * lddb + tx] : dB[ty * lddb + tx] ;
  }
  __syncthreads();
    
  device_gemm_batched_smallsq_kernel(N, alpha, sA, slda, sB, sldb, beta, dC, lddc);
    
    return;
}



__global__ void
parasof_That_gemm_batched_kernel(int N,
				 tipo const * const * dA_array, int ldda, 
				 tipo**  dT_array, int lddt, 
				 tipo_int batchCount)
{

  /* dC = alpha*op(dA)*op(dB) + beta(dC) */
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,batchCount -1

  
  tipo_int batchid = bx * blockDim.z + tz; // = bx per ogni blocco

  const tipo * __restrict__ dA;
  const tipo * __restrict__ dB;
  tipo* __restrict__ dC;
  
  const int slda = SLDA(N); 
  const int sldb = SLDA(N);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + blockDim.z * slda * N);

  sA += tz * slda * N;
  sB += tz * sldb * N;
  
  if(batchid >= batchCount) return;

  dA = dA_array[batchid] + 0 * ldda + 0; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[batchid] + 0 * ldda + 0; // dB = dB_array[batchid][bi:,bj:]
  dC = dT_array[batchid] + 0 * lddt + 0;  // dC = dC_array[batchid][ci:,cj:]
    
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
    
  __syncthreads();
    
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_zero, dC, lddt);
  __syncthreads();
    
  dA = dA_array[batchid] + 0 * ldda + N; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[batchid] + 0 * ldda + N; // dB = dB_array[batchid][bi:,bj:]
    
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
  __syncthreads();
    
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_one, dC, lddt);
  __syncthreads();
    
  return;
}

__global__ void
sof_That_gemm_batched_kernel(int N,
			     tipo const * const * dA_array, int ldda, 
			     tipo**  dT_array, int lddt, 
			     int nb_slices,
			     int slice_size,
			     int offset,
			     tipo_int batchCount)
{ 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,nb_slices -1
  
  int sliceId = bx * blockDim.z + tz; // = bx per ogni blocco
  if (sliceId >= nb_slices)
    return;
  
  const tipo * __restrict__ dA;
  const tipo * __restrict__ dB;
  tipo* __restrict__ dC;
  
  const int slda = SLDA(N); 
  const int sldb = SLDA(N);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + blockDim.z * slda * N);
  sA += tz * N * slda;
  sB += tz * N * sldb;
  
  if (sliceId >= nb_slices) return;

  dA = dA_array[sliceId] + 0 * ldda + 0; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[sliceId] + 0 * ldda + 0; // dB = dB_array[batchid][bi:,bj:]
  dC = dT_array[sliceId] + 0 * lddt + 0;  // dC = dC_array[batchid][ci:,cj:]
  
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
  
  __syncthreads();
  
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_zero, dC, lddt);
  __syncthreads();
  
  dA = dA_array[sliceId] + 0 * ldda + N; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[sliceId] + 0 * ldda + N; // dB = dB_array[batchid][bi:,bj:]
  
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
  __syncthreads();
  
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_one, dC, lddt);
  __syncthreads();
  return;
}

__global__ void
sof_That_gemm_batched_kernel_FACT(int N,
				  tipo const * const * dA_array, int ldda, 
				  tipo**  dT_array, int lddt, 
				  int nb_slices,
				  int slice_size,
				  int offset,
				  tipo_int batchCount)
{ 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,nb_slices -1
  
  int sliceId = bx * blockDim.z + tz; // = bx per ogni blocco
  int batchId_A = sliceId * slice_size + offset;
  if ((sliceId >= nb_slices) || (batchId_A >= batchCount))
    return;

  
  const tipo * __restrict__ dA;
  const tipo * __restrict__ dB;
  tipo* __restrict__ dC;
  
  const int slda = SLDA(N); 
  const int sldb = SLDA(N);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + blockDim.z * slda * N);
  sA += tz * N * slda;
  sB += tz * N * sldb;
  
  if (sliceId >= nb_slices) return;

  dA = dA_array[batchId_A] + 0 * ldda + 0; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[batchId_A] + 0 * ldda + 0; // dB = dB_array[batchid][bi:,bj:]
  dC = dT_array[batchId_A] + 0 * lddt + 0;  // dC = dC_array[batchid][ci:,cj:]
  
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
  
  __syncthreads();
  
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_zero, dC, lddt);
  __syncthreads();
  
  dA = dA_array[batchId_A] + 0 * ldda + N; // dA = dA_array[batchid][ai:,aj:]
  dB = dA_array[batchId_A] + 0 * ldda + N; // dB = dB_array[batchid][bi:,bj:]
  
  // read A & B 
  sA[tx * slda + ty] = dA[ty * ldda + tx];
  sB[ty * sldb + tx] = dB[ty * ldda + tx];
  __syncthreads();
  
  device_gemm_batched_smallsq_kernel(N, d_one, sA, slda, sB, sldb, d_one, dC, lddt);
  __syncthreads();
  return;
}

extern "C" void 
magmablas_gemm_batched_smallsq(int itransA, int itransB,
			       int m, int n, int k,
			       //tipo alpha,
			       tipo **dA_array, int ai, int aj, int ldda, 
			       tipo **dB_array, int bi, int bj, int lddb,
			       //tipo beta,
			       tipo **dC_array, int ci, int cj, int lddc, 
			       tipo_int batchCount )
{ 
  if( !(m == n  && n == k) ){
    printf("Only square sizes are supported\n");
    return;
  }
  
  if( m > 32){
    printf("Only square sizes of up to 32 are supported\n");
    return;
    }
  
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  
  int ntcol  = 1; //magma_get_dgemm_batched_ntcol( m ); // numero di matrici trocessate da un blocco di threads
  int shmem  = ( SLDA(m)*m + SLDA(n)*n ) * sizeof(tipo); // per caricare le matrici A e B in shared
  shmem *= ntcol;
  
  //const int nblocks = magma_ceildiv(batchCount, ntcol);
  tipo_int nblocks = batchCount;
  dim3 grid(1, 1, nblocks);
  dim3 threads(m, m, ntcol);

  tipo alpha = one, beta = zero;

  clock_t start = clock();
  gemm_batched_smallsq_kernel<<<grid, threads, shmem >>>(itransA, itransB, m, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, 
							 batchCount);
  cudaDeviceSynchronize();
  clock_t stop = clock();
  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;
  printf("tempo gpu mio %f \n", time);
  
  return;
}

extern "C" void 
cublas_gemm_batched(int itransA, int itransB,
		    int m, int n, int k,
		    //tipo alpha,
		    tipo **dA_array, int lda, 
		    tipo **dB_array, int ldb,
		    //tipo beta,
		    tipo **dC_array, int ldc, 
		    tipo_int batchCount )
{ 
  if( !(m == n  && n == k) ){
    printf("Only square sizes are supported\n");
    return;
  }
  
  if( m > 32){
    printf("Only square sizes of up to 32 are supported\n");
    return;
    }
  
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  cublasOperation_t transA = (itransA == 0) ? cublas_trans : cublas_trans_t;
  cublasOperation_t transB = (itransB == 0) ? cublas_trans : cublas_trans_t;

  tipo alpha = one, beta = zero;

  clock_t start = clock();

#ifdef DOUBLE_PRECISION
  cublasStatus1 = cublasDgemmBatched(cublasHandle, transA, transB,
				     int(m), int(n), int(k),
				     &alpha,
				     dA_array, int(lda),
				     dB_array, int(ldb),
				     &beta, dC_array, int(ldc), batchCount );
#else
  cublasStatus1 = cublasSgemmBatched(cublasHandle, transA, transB,
				     int(m), int(n), int(k),
				     &alpha,
				     dA_array, int(lda),
				     dB_array, int(ldb),
				     &beta, dC_array, int(ldc), batchCount );
#endif
  cudaDeviceSynchronize();
  clock_t stop = clock();
  if (cublasStatus1 != CUBLAS_STATUS_SUCCESS){
    printf("ERROR: cublas<t>gemmBatched failed \n");
    return;
  }

  tipo time = ((tipo) (stop - start)) / CLOCKS_PER_SEC;
  printf("tempo cublas %f \n", time);
  
  return;
}



__device__ void load_2d_shared(cublasOperation_t transA, tipo* dA, int ai, int aj, int ldda, tipo* sA, int slda){
  tipo* dA_tmp = dA + aj * ldda + ai; // dA = dA_array[batchid][ai:,aj:]
  
  // read A 
  if(transA == d_cublas_trans){
    sA[threadIdx.y * slda + threadIdx.x] = dA_tmp[threadIdx.x + threadIdx.y * ldda];
  }
  else{
    sA[threadIdx.x * slda + threadIdx.y] = (transA == d_cublas_trans_t) ? dA_tmp[threadIdx.y * ldda + threadIdx.x] : dA_tmp[threadIdx.y * ldda + threadIdx.x] ;
  }
  return;
}

__device__ void load_1d_shared(tipo* dx, int incx, tipo* sx, int sincx){

  if (threadIdx.y == 0){  
    sx[threadIdx.x * sincx] = dx[threadIdx.x * incx];
  }
  return;
}


/******************************************************************************/
__global__ void gemm_smallsq_kernel(cublasOperation_t transA, cublasOperation_t transB,
				    int n,
				    tipo* dA, int lda,
				    tipo* dB, int ldb,
				    tipo* dC, int ldc){

  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + slda * n);
  
  tipo rTmp = d_zero;

  load_2d_shared(transA, dA, 0, 0, lda, sA, slda);
  load_2d_shared(transB, dB, 0, 0, ldb, sB, sldb);
  __syncthreads();
  
  // multiply
#pragma unroll
  for(int j = 0; j < n; j++){
    rTmp += sA[j * slda + threadIdx.x] * sB[threadIdx.y * sldb + j]; //  sA[threadIdx.x, j]*sB[j, threadIdx.y]
  }
  // write from rTmp
  dC[threadIdx.y * ldc + threadIdx.x] = rTmp;

  return;
}

/******************************************************************************/
static __device__ void  
gemv1d_batched_smallsq_device(cublasOperation_t transA, int n, tipo alpha, tipo* A, int lda, tipo* sx, int incx, tipo beta, tipo* y, int incy)
{
  /* Calcola y = alpha*A*x + beta*y
   assume che sx sia in shared memory */
  
  // threads are all configurated locally
  int tx = threadIdx.x;

  if ((n <= 0) || (tx >= n)) return;
  
  tipo tmp = d_zero;

  if (transA == d_cublas_trans){
    for(int i = 0; i<n; i++)
      tmp+= A[tx + i * lda]*sx[i*incx];
  }else if (transA == d_cublas_trans_t){
    for(int i = 0; i<n; i++){
      tmp+= A[i + tx * lda]*sx[i*incx];
    }
  }
  
  y[tx*incy] = alpha * tmp + beta * y[tx*incy];
  
  return;
}


/******************************************************************************/
static __device__ void 
gemv2d_batched_smallsq_device(int n, tipo alpha, tipo* sA, int slda, tipo* sx, int incx, tipo beta, tipo* sy, int incy)
{
  /* Calcola y = alpha*A*x + beta*y
   assume che sA, sx  siano in shared memory
   Attenzione: modifica i valori di A */
  
  // threads are all configurated locally
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  if (n <= 0) return; //((n <= 0) || (tx >= n) || (ty >= n))
  
  sA[tx + ty * slda] =  sA[tx + ty * slda]*sx[ty * incx];
  __syncthreads();

  if (ty == 0){
    for(int i = 1; i<n; i++)
      sA[tx]+= sA[tx + i * slda];
    
    if (tx < n)
      sy[tx*incy] = alpha * sA[tx] + beta * sy[tx*incy];
  }
  
  return;
}

__global__ void
parasof_larfb_gemv_batched_smallsq_kernel(int m, int n,
					  tipo** dV_array, int lddv, 
					  tipo** dT_array, int lddt, 
					  tipo** db_array, tipo** dbout_array,
					  tipo_int batchCount,
					  tipo_int offset)
{ 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  //const int tz = threadIdx.z; // = 0 per ogni thread
  const int bx = blockIdx.z;

  //const int batchid = bx * blockDim.z + tz; // = bx per ogni blocco
  tipo_int batchid = bx; // = bx per ogni blocco
  int lx, vi, vj, ti = 0, tj = 0;
  tipo * dV; 
  tipo * dT; 
  tipo * db;
  const int slda = SLDA(n); 
  tipo* sA = (tipo*)(zdata);
  tipo* sx = (tipo*)(zdata + blockDim.z * slda * n);
  tipo* sy = (tipo*)(zdata + blockDim.z * slda * n + n);
  
  if(batchid >= batchCount) return;
  
  dV = dV_array[batchid]; 
  dT = dT_array[batchid]; 
  db = db_array[batchid];
    
  // sA += tz * slda * n;
  // sx += tz * n;
  // sy += tz * n;
  
  // Applico a sx
  //scrivo nel blocco 2
  
  if (batchid < offset){
    lx = batchCount + batchid - offset;
    //load_1d_shared(db_array[lx], 1, sx, 1);
  }else{
    lx = batchid - offset;
    //load_1d_shared(db, 1, sx, 1);
  }
  load_1d_shared(db, 1, sx, 1);
  
  //CARICO IN SHARED:
  vi = n, vj = 0;
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V2.T
  __syncthreads();
  
  //sy = sA*sx = V2.T*b2;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  vi = 0, vj = 0;
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V1.T
  load_1d_shared(db_array[lx], 1, sx, 1);
  __syncthreads();
  
  //sy = sA*sx + sy = V1.T*b1 + V2.T*b2;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_one, sy, 1);

  //CARICO IN SHARED:
  __syncthreads();
  
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sx = sA*sy = T.T*(V1.T*b1 + V2.T*b2);
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sy, 1, d_zero, sx, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  load_1d_shared(db, 1, sy, 1);
  __syncthreads();

  //sy = sA*sx + sy = b - V2*T.T*(V1.T*b1 + V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_oneopp, sA, slda, sx, 1, d_one, sy, 1);
  __syncthreads();
  
  //write from sy
  db = dbout_array[lx];
  if (ty == 0)
    db[tx] = sy[tx];


}


__global__ void
sof_larfb_gemv_batched_smallsq_kernel(int m, int n,
				      tipo** dV_array, int lddv, 
				      tipo** dT_array, int lddt, 
				      tipo** db_array,
				      int nb_slices, int slice_size, int offset,  tipo_int batchCount)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  const int bx = blockIdx.z;
  int sliceid = bx;
  
  tipo_int batchid = bx * slice_size + offset; 

  tipo * dV; 
  tipo * dT; 
  tipo * db;
  const int slda = SLDA(n); 
  tipo* sA = (tipo*)(zdata);
  tipo* sx = (tipo*)(zdata + blockDim.z * slda * n);
  tipo* sy = (tipo*)(zdata + blockDim.z * slda * n + n);
  
  if ((batchid >= batchCount) || (sliceid >= nb_slices)) return;
  
  dV = dV_array[sliceid]; 
  dT = dT_array[sliceid]; 
  db = db_array[batchid-1];
     
  // sA += tz * slda * n;
  // sx += tz * n;
  // sy += tz * n;
  
  // Applico a sx
  //scrivo nel blocco 2
  int vi = 0, vj = 0, ti = 0, tj = 0;
  load_1d_shared(db, 1, sx, 1);

  // if ((tx == 0) && (ty == 0)){
  //   printf("batchid %d: %f %f  \n", batchid-1, db[0], db[1] );
  // }
  // __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V1.T
  __syncthreads();
  
  //sy = sA*sx = V1.T*b1;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  vi = n, vj = 0;
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V2.T
  load_1d_shared(db_array[batchid], 1, sx, 1); // sx = sb[batchid]
  __syncthreads();
  
  //sy = sA*sx + sy = V1.T*b1 + V2.T*b2;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_one, sy, 1);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sx = sA*sy = T.T*(V1.T*b1 + V2.T*b2);
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sy, 1, d_zero, sx, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  
  //sy = sA*sx + db = db - V1*T.T*(V1.T*b1 + V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_oneopp, sA, slda, sx, 1, d_one, db, 1);
  __syncthreads();

  // if ((tx == 0) && (ty == 0)){
  //   printf("batchid %d: %f %f  \n", batchid-1, db[0], db[1] );
  // }
  // __syncthreads();
  
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  
  //sy = sA*sx + db = db - V2*T.T*(V1.T*b1 + V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_oneopp, sA, slda, sx, 1, d_one, db_array[batchid], 1);
  __syncthreads();
  
}

__global__ void
sof_larfb_gemv_batched_smallsq_kernel_FACT(int m, int n,
					   tipo** dV_array, int lddv, 
					   tipo** dT_array, int lddt, 
					   tipo** db_array,
					   int nb_slices, int slice_size, int offset,  tipo_int batchCount)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  const int bx = blockIdx.z;
  int sliceid = bx;
  
  tipo_int batchid = bx * slice_size + offset; 

  tipo * dV; 
  tipo * dT; 
  tipo * db;
  const int slda = SLDA(n); 
  tipo* sA = (tipo*)(zdata);
  tipo* sx = (tipo*)(zdata + blockDim.z * slda * n);
  tipo* sy = (tipo*)(zdata + blockDim.z * slda * n + n);
  
  if ((batchid >= batchCount) || (sliceid >= nb_slices)) return;
  
  dV = dV_array[batchid]; 
  dT = dT_array[batchid]; 
  db = db_array[batchid-1];
     
  
  // Applico a sx
  //scrivo nel blocco 2
  int vi = 0, vj = 0, ti = 0, tj = 0;
  load_1d_shared(db, 1, sx, 1);

  // if ((tx == 0) && (ty == 0)){
  //   printf("batchid %d: %f %f  \n", batchid-1, db[0], db[1] );
  // }
  // __syncthreads();
   // if ((tx == 0) && (ty == 0) && (batchid == 1)){
  //   printf("V2.T: \n %f %f \n %f %f  \n", sA[0*slda+0], sA[1*slda+0], sA[0*slda+1], sA[1*slda+1] );
  //   printf("T.T: \n %f %f \n %f %f  \n",sB[0*sldb+0], sB[1*sldb+0], sB[0*sldb+1], sB[1*sldb+1]  );
  // }
  // __syncthreads(); 
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V1.T
  __syncthreads();
  
  //sy = sA*sx = V1.T*b1;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  vi = n, vj = 0;
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V2.T
  load_1d_shared(db_array[batchid], 1, sx, 1); // sx = sb[batchid]
  __syncthreads();
  
  //sy = sA*sx + sy = V1.T*b1 + V2.T*b2;
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_one, sy, 1);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sx = sA*sy = T.T*(V1.T*b1 + V2.T*b2);
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sy, 1, d_zero, sx, 1);
  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  
  //sy = sA*sx + db = b1 - V1*T.T*(V1.T*b1 + V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_oneopp, sA, slda, sx, 1, d_one, db, 1);
  __syncthreads();

  // if ((tx == 0) && (ty == 0)){
  //   printf("batchid %d: %f %f  \n", batchid-1, db[0], db[1] );
  // }
  // __syncthreads();
  
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  
  //sy = sA*sx + db = b2 - V2*T.T*(V1.T*b1 + V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_oneopp, sA, slda, sx, 1, d_one, db_array[batchid], 1);
  __syncthreads();
  
}



__global__ void
parasof_larfb_gemm_batched_smallsq_kernel(int m, int n,
					  tipo** dV_array, int lddv, 
					  tipo** dT_array, int lddt, 
					  tipo** dA_array, int ldda,
					  tipo_int batchCount,
					  tipo_int offset)
  {
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  //const int tz = threadIdx.z; // = 0 per ogni thread
  const int bx = blockIdx.z;
  
  int vi, vj, ti = 0, tj = 0, ai, aj;
  int sx, dx;
  tipo * dV; 
  tipo * dT; 
  tipo * dA;
	     
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata +   blockDim.z * slda * n);
  tipo* sC = (tipo*)(zdata + 2*blockDim.z * slda * n);
  
  //const int batchid = bx * blockDim.z + tz; // = bx per ogni blocco
  tipo_int batchid = bx;

  if (batchid >= batchCount) return;
  
  dV = dV_array[batchid]; 
  dT = dT_array[batchid]; 
  
  //sA += tz * slda * n;
  //sB += tz * sldb * n;
  //sC += tz * sldc * n;
  
  // Applico a sx
  //scrivo nel blocco 2
  ai = n; aj = 0;
  if (batchid < offset){
    sx = batchCount + batchid - offset;
  }else{
    sx = batchid - offset;
  }
  vi = 0, vj = 0; 

  dA = dA_array[sx];
  
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sB, sldb); //sA = T.T
  //load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A
  __syncthreads();
  
  //sC = sB*sA = T.T*V.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc); 

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  __syncthreads();
  
  //sB = sA*sC = V*(T.T*V.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED:
  __syncthreads();
  //load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sA, slda); //sA = A
  __syncthreads();

  //sC = sB*sA = (V*T.T*V.T)*A
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx];
  
  /*********************************************************************************/
  // Applico a dx
  //scrivo nel blocco 1
  ai = 0; aj = 0;
  dx = (batchid + offset) %  batchCount;
  vi = n, vj = 0;
  
  dA = dA_array[dx];
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  //load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sB, sldb); //sA = T.T
  __syncthreads();
  //sC = sB*sA = T.T*V.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);

  //CARICO IN SHARED:
  __syncthreads();
  //load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  __syncthreads();
  //sB = sA*sC = V*(T.T*V.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED
  __syncthreads();
  //load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sA, slda); //sA = A
  __syncthreads();
  //sC = sB*sA = (V*T.T*V.T)*A
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx];
  
  return;
}


__device__ void device_scrivi_blocco_csr(int n, tipo* sA, int slda,
					 tipo *csrValM, tipo *csrColIndM, 
					 //tipo * csrValMptr, tipo * csrColIndMptr, int it,
					 int block_row_idx, int block, int m_id,
					 tipo_int batchid, tipo_int offset){
  /* block = 0 se blocco sx, 1 se blocco dx
     block_row_idx = indice blocco riga
     thread (tx,ty) scrive l'elemento (tx,ty) di sA su csrValM[ block_row_idx*(2*n*n) + block*n + tx*2*n + ty]
  */

  int idx, idcol;

  
  if ((threadIdx.x < n) && (threadIdx.y  < n)){
    // ho trasposto gli indici cosi da rendere gli accessi in memoria allineati
    idx = (block_row_idx*n + threadIdx.y)*2*n + block*n + threadIdx.x; //  csrRowPtrM[block_row_idx*n + threadIdx.x] + block*n + threadIdx.y;
    
    if (m_id && (threadIdx.x == threadIdx.y)){
      csrValM[ idx ]    =  d_one - sA[threadIdx.x * slda + threadIdx.y];
      //csrValMptr[ idx ] =  d_one - sA[threadIdx.y * slda + threadIdx.x];
    }else{
      csrValM[ idx ]    =  d_oneopp*sA[threadIdx.x * slda + threadIdx.y];
      //csrValMptr[ idx ] =  d_oneopp*sA[threadIdx.y * slda + threadIdx.x];
    }

    if ((batchid < offset) && (block == 0))
      idcol = batchid;
    else if ((batchid < offset) && (block == 1))
      idcol = block_row_idx;
    else if (block == 0)
      idcol = block_row_idx;
    else if (block == 1)
      idcol = batchid;
    
    csrColIndM[ idx ]    = idcol*n + threadIdx.x;
    //csrColIndMptr[ idx ] = idcol*n + threadIdx.y;

  }
  
  return;
}


__global__ void
parasof_larfb_calcolaM_gemm_batched_smallsq_kernel(int m, int n,
						   tipo** dV_array, int lddv, 
						   tipo** dT_array, int lddt, 
						   tipo** dA_array, int ldda,
						   //tipo* csrValM, int* csrColIndM, 
						   tipo** csrValMptr, tipo** csrColIndMptr, int it,				   
						   tipo_int batchCount,
						   tipo_int offset)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  //const int tz = threadIdx.z; // = 0 per ogni thread
  const int bx = blockIdx.z;
  
  int vi, vj, ti = 0, tj = 0, ai, aj;
  int sx, dx, block;
  tipo * dV; 
  tipo * dT; 
  tipo * dA;

	     
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata +   blockDim.z * slda * n);
  tipo* sC = (tipo*)(zdata + 2*blockDim.z * slda * n);
  
  //const int batchid = bx * blockDim.z + tz; // = bx per ogni blocco
  tipo_int batchid = bx;

  if (batchid >= batchCount) return;
  
  dV = dV_array[batchid]; 
  dT = dT_array[batchid];
  
  // tipo * csrVal;
  // int * csrColInd;
  // csrVal = csrValM[it];
  // csrColInd = csrColIndM[it];
  
  //sA += tz * slda * n;
  //sB += tz * sldb * n;
  //sC += tz * sldc * n;
  
  // Applico a sx
  //scrivo nel blocco 2
  ai = n; aj = 0;
  if (batchid < offset){
    sx = batchCount + batchid - offset;
    block = 1; 
  }else{
    sx = batchid - offset;
    block = 0; 
  }
  vi = 0, vj = 0; 

  dA = dA_array[sx];
  
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sB, sldb); //sA = T.T
  //load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A
  __syncthreads();
  
  //sC = sB*sA = T.T*V.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc); 

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  __syncthreads();
  
  //sB = sA*sC = V*(T.T*V.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  
  //CARICO IN SHARED:
  __syncthreads();
  device_scrivi_blocco_csr(n, sB, sldb,
			   //csrValM, csrColIndM,
			   csrValMptr[it], csrColIndMptr[it], 
			   sx, block, 0,
			   batchid, offset);
  
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sA, slda); //sA = A
  __syncthreads();

  //sC = sB*sA = (V*T.T*V.T)*A
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx]; // A = - (V*T.T*V.T)*A
  
  /*********************************************************************************/
  // Applico a dx
  //scrivo nel blocco 1

  ai = 0; aj = 0;
  if (batchid < offset)
    block = 0; 
  else
    block = 1; 
  dx = (batchid + offset) %  batchCount;
  vi = n, vj = 0;
  
  dA = dA_array[dx];
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  //load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sB, sldb); //sA = T.T
  __syncthreads();
  //sC = sB*sA = T.T*V.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);

  //CARICO IN SHARED:
  __syncthreads();
  //load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  __syncthreads();
  //sB = sA*sC = V*(T.T*V.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED
  __syncthreads();
  device_scrivi_blocco_csr(n, sB, sldb,
			   //csrValM, csrColIndM, 
			   csrValMptr[it], csrColIndMptr[it], 
			   sx, block, 1, // 0 o 1?
			   batchid, offset);
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sA, slda); //sA = A
  __syncthreads();
  //sC = sB*sA = (V*T.T*V.T)*A
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx]; // A = A - sC = A -  (V*T.T*V.T)*A
  
  return;
}


__global__ void
sof_larfb_gemm_batched_smallsq_kernel(int m, int n,
				      tipo** dV_array, int lddv, 
				      tipo** dT_array, int lddt, 
				      tipo** dA_array, int ldda,
				      int nb_slices, int slice_size, int offset,  tipo_int batchCount)
{ 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.z;

  tipo_int batchid = bx*slice_size + offset;
  int sliceid = bx;
  
  int vi, vj, ti = 0, tj = 0, ai, aj;
  int sx, dx;
  tipo * dV; 
  tipo * dT; 
  tipo * dA;
	     
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata +   blockDim.z * slda * n);
  tipo* sC = (tipo*)(zdata + 2*blockDim.z * slda * n);
  
  if ((batchid >= batchCount) || (sliceid >= nb_slices)) return;

  dV = dV_array[sliceid]; 
  dT = dT_array[sliceid]; 
  
  sx = batchid - 1;
  dx = batchid + 1;

  ai = n; aj = 0;
  vi = 0, vj = 0; // V1
  dA = dA_array[sx]; // S

 
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = S
  __syncthreads();
  
  //sC = sA*sB = V.T*A = V1.T*S
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc); 

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sB = sA*sC = T.T*(V.T*A) = T.T*(V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();

  //sC = sA*sB = V*(T.T*V.T*A) = V1*(T.T*V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx];
  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();

  //sC = sA*sB = V*(T.T*V.T*A) = V2*(T.T*V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  ai = n; aj = 0; 
  dA = dA_array[batchid]; // S
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx];
  
  /*********************************************************************************/
  // Applico a dx: il blocco non zero e' il blocco 1 in alto
  //scrivo nel blocco 1
  ai = 0; aj = 0;
  vi = n, vj = 0; // V2
  dA = dA_array[dx];
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V2.T
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A = T
  __syncthreads();
  //sC = sA*sB = V.T*A = V2.T*T
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  //sB = sA*sC = T.T*(V.T*A) = T.T(V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();
  //sC = sA*sB = V*(T.T*V.T*A) = V2*(T.T*V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx];

  //CARICO IN SHARED
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  //sC = sA*sB = V*(T.T*V.T*A) = V1*(T.T*V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  ai = 0; aj = 0;
  dA = dA_array[batchid];
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx];

  return;
}

__global__ void
sof_larfb_gemm_batched_smallsq_kernel_FACT(int m, int n,
					   tipo** dV_array, int lddv, 
					   tipo** dT_array, int lddt, 
					   tipo** dA_array, int ldda,
					   int nb_slices, int slice_size, int offset,  tipo_int batchCount)
{ 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.z;

  tipo_int batchid = bx*slice_size + offset;
  int sliceid = bx;
  
  int vi, vj, ti = 0, tj = 0, ai, aj;
  int sx, dx;
  tipo * dV; 
  tipo * dT; 
  tipo * dA;
	     
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata +   blockDim.z * slda * n);
  tipo* sC = (tipo*)(zdata + 2*blockDim.z * slda * n);
  
  if ((batchid >= batchCount) || (sliceid >= nb_slices)) return;

  dV = dV_array[batchid]; 
  dT = dT_array[batchid]; 
  
  sx = batchid - 1;
  dx = batchid + 1;

  ai = n; aj = 0;
  vi = 0, vj = 0; // V1
  dA = dA_array[sx]; // S

 
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V.T
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = S
  __syncthreads();
  
  //sC = sA*sB = V.T*A = V1.T*S
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc); 

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sB = sA*sC = T.T*(V.T*A) = T.T*(V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();

  //sC = sA*sB = V*(T.T*V.T*A) = V1*(T.T*V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx];
  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();

  //sC = sA*sB = V*(T.T*V.T*A) = V2*(T.T*V1.T*S)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  ai = n; aj = 0; 
  dA = dA_array[batchid]; // S
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx];
  
  /*********************************************************************************/
  // Applico a dx: il blocco non zero e' il blocco 1 in alto
  //scrivo nel blocco 1
  ai = 0; aj = 0;
  vi = n, vj = 0; // V2
  dA = dA_array[dx];
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, vi, vj, lddv, sA, slda); //sA = V2.T
  load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A = T
  __syncthreads();
  //sC = sA*sB = V.T*A = V2.T*T
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);

  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  __syncthreads();
  //sB = sA*sC = T.T*(V.T*A) = T.T(V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  //CARICO IN SHARED
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();
  //sC = sA*sB = V*(T.T*V.T*A) = V2*(T.T*V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] -= sC[ty * sldc + tx];

  //CARICO IN SHARED
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();
  //sC = sA*sB = V*(T.T*V.T*A) = V1*(T.T*V2.T*T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  ai = 0; aj = 0;
  dA = dA_array[batchid];
  
  //write from sC
  dA += aj * ldda + ai;  // dC = dC_array[batchid][ci:,cj:]
  dA[ty * ldda + tx] = d_oneopp*sC[ty * sldc + tx];

  return;
}


__global__ void larfb_step1_last_call_kernel(int m, int n,
					     tipo** dV_array, int lddv, 
					     tipo** dT_array, int lddt,
					     // tipo** dA_array, int ldda, // per scrivere A1 direttamente su Aout 
					     tipo **b_array,
					     tipo_int batchCount, tipo_int offset){
  /* kernel che aggiorna il sistema lineare dopo la prima QR nella risoluzione dei sistemi lineari 2x2
     necessita di memoria shared dimanica
   */
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int batchid  = blockIdx.x;
   
  if ((tx >= n) || (ty >= n) || (batchid > (batchCount/2))) return;
  
  tipo *dV = dV_array[batchid]; // dV = [[V1],[V2]]
  tipo *dT = dT_array[batchid]; 
  tipo *dA = dV_array[batchid + offset]; // dA = [[A1],[A2]]
  tipo *db;

  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + slda * n);
  tipo* sC = (tipo*)(zdata + 2 * slda * n);
  tipo* sx = (tipo*)(zdata + 3 * slda * n );
  tipo* sy = (tipo*)(zdata + 3 * slda * n +  n);
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, n, 0, lddv, sA, slda); // V2.T 
  load_2d_shared(d_cublas_trans,   dA, n, 0, lddv, sB, sldb); // A2
  if (ty == 0)
    load_1d_shared(b_array[batchid + offset],  1, sx, 1); //b2
  __syncthreads();
    
  //sC = sA*sB = V2.T@A2
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  //sy = sA*sx = V2.T*b2
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, 0, 0, lddv, sA, slda); // V1.T
  load_2d_shared(d_cublas_trans,   dA, 0, 0, lddv, sB, sldb); // A1
  if (ty == 0)
    load_1d_shared(b_array[batchid],  1, sx, 1); //b1
  __syncthreads();
  
  //sC += sA*sB = V1.T*A1 + (V2.T*A2)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_one, sC, sldc);
  __syncthreads();
  //sy += sA*sx = V1.T*b1 + (V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_one, sy, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dT, 0, 0, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sB = sA*sC = T.T*(V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();
  
  //sx = sA*sy = T.T*(V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sy, 1, d_zero, sx, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();

  //sC = sA*sB = V1*(T.T*V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  //sy = sA*sx = V1*(T.T*V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();

  //scrivo i risultati in global memory
  //dA = dA_array[1];  // scrivo il blocco (1,2) in Aout cosi' e' gia' pronto per il sistema triangolare
  //dA[ty * ldda + tx] -= sC[ty * sldc + tx];
  dA[ty * lddv + tx] -= sC[ty * sldc + tx];
  db = b_array[batchid];
  if (ty == 0)
    db[tx] -= sy[tx];

  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();
  
  //sC = sA*sB = V2*(T.T*V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  //sy = sA*sx = V2*(T.T*V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();
  
  //scrivo i risultati in global memory
  //dA = dV_array[1] + 0 * lddv + n; // scrivo il blocco (2,2) in Ain perche' poi ne faccio la QR
  dA +=  0 * lddv + n; // scrivo il blocco (2,2) in Ain perche' poi ne faccio la QR 
  dA[ty * lddv + tx] -= sC[ty * sldc + tx];
  if (ty == 0)
    b_array[batchid + offset][tx] -= sy[tx];

  
  return;
}


__global__ void larfb_calcolaM_step1_last_call_kernel(int m, int n,
						       tipo** dV_array, int lddv, 
						       tipo** dT_array, int lddt,
						       tipo** csrValMptr, tipo** csrColIndMptr, int it,
						       tipo **b_array,
						       tipo_int batchCount, tipo_int offset){
 /* kernel che aggiorna il sistema lineare dopo la prima QR nella risoluzione dei sistemi lineari 2x2
     necessita di memoria shared dimanica
   */
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int batchid  = blockIdx.x;
  int idx, m_id, idcol;
  
  if ((tx >= n) || (ty >= n) || (batchid > offset)) return; //(batchid > (batchCount/2))
  
  tipo *dV = dV_array[batchid]; // dV = [[V1],[V2]]
  tipo *dT = dT_array[batchid]; 
  tipo *dA = dV_array[batchid + offset]; // dA = [[A1],[A2]]
  tipo *db;
  tipo *csrValM = csrValMptr[it], *csrColIndM = csrColIndMptr[it];
    
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + slda * n);
  tipo* sC = (tipo*)(zdata + 2 * slda * n);
  tipo* sx = (tipo*)(zdata + 3 * slda * n );
  tipo* sy = (tipo*)(zdata + 3 * slda * n +  n);
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, n, 0, lddv, sA, slda); // sA = V2.T 
  load_2d_shared(d_cublas_trans_t, dT, 0, 0, lddt, sB, sldb); // sB = T.T
  __syncthreads();

  // if ((tx == 0) && (ty == 0) && (batchid == 1)){
  //   printf("V2.T: \n %f %f \n %f %f  \n", sA[0*slda+0], sA[1*slda+0], sA[0*slda+1], sA[1*slda+1] );
  //   printf("T.T: \n %f %f \n %f %f  \n",sB[0*sldb+0], sB[1*sldb+0], sB[0*sldb+1], sB[1*sldb+1]  );
  // }
  // __syncthreads(); 
    
  //sC = sB*sA = T.T@V2.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();

  // if ((tx == 0) && (ty == 0) && (batchid == 1)){
  //   printf("T.T*V2.T: \n %f %f \n %f %f  \n", sC[0*sldc+0], sC[1*sldc+0], sC[0*sldc+1], sC[1*sldc+1] );
  // }

  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); // sA = V1
  __syncthreads();

  
  // if ((tx == 0) && (ty == 0) && (batchid == 1)){
  //   printf("V1: \n %f %f \n %f %f  \n",  sA[1*slda+0], sA[0*slda+1], sA[1*slda+1] );
  // }

  //sB = sA*sC = V1*(T.T*V2.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();

  // if ((tx == 0) && (ty == 0) && (batchid == 1))
  //   printf("V1*T.T*V2.T: \n %f %f \n %f %f  \n", sB[0*sldb+0], sB[1*sldb+0], sB[0*sldb+1], sB[1*sldb+1] );
  // __syncthreads(); 
  
  //BLOCCO 12
  if (0){
    device_scrivi_blocco_csr(n, sB, sldb,
			     //csrValM, csrColIndM,
			     csrValMptr[it], csrColIndMptr[it], 
			     batchid, 1, 0,
			     batchid+offset,  offset);
  }else{
    idx = (batchid*n + ty)*2*n + 1*n + tx; //  csrRowPtrM[block_row_idx*n + threadIdx.x] + block*n + threadIdx.y;
    m_id = 0;
    if (m_id && (tx == ty)){
      csrValM[ idx ]    =  d_one - sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_one - sA[threadIdx.y * slda + threadIdx.x];
    }else{
      csrValM[ idx ]    =  d_oneopp*sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_oneopp*sA[threadIdx.y * slda + threadIdx.x];
    }
    idcol = batchid+offset;   
    csrColIndM[ idx ]    = idcol*n + threadIdx.x;
  }
  __syncthreads();

  // if ((tx == 0) && (ty == 0) && (batchid == 1))
  //   printf("idx %d, idcol %d, csrValM[0,8] =  %f, csrValM[1,8] =  %f  \n", idx, idcol, csrValM[ (batchid*n + 0)*2*n + 1*n + 0], csrValM[ (batchid*n + 0)*2*n + 1*n + 1] );
  // if ((batchid == 1) && (tx == 0) && (ty == 0))
  //   printf(" csrValM[%d,%d]  = %f  \n", ty, tx, csrValM[ (batchid*n + 0)*2*n + 1*n + ty] );
  // if ((batchid == 1) && (tx == 0) && (ty == 1))
  //   printf(" csrValM[%d,%d]  = %f  \n", ty, tx, csrValM[ (batchid*n + 0)*2*n + 1*n + ty] );
  // __syncthreads(); 

  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); // sA = V2
  __syncthreads();
  
  //sB = sA*sC = V2*T.T*V2.T
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();

  //BLOCCO 22
  if(0){
    device_scrivi_blocco_csr(n, sB, sldb,
			     //csrValM, csrColIndM,
			     csrValMptr[it], csrColIndMptr[it], 
			     batchid+offset, 1, 1,
			     batchid,  offset);
  }else{
    idx = ((batchid+offset)*n + ty)*2*n + 1*n + tx; //  csrRowPtrM[block_row_idx*n + threadIdx.x] + block*n + threadIdx.y;
    m_id = 1;
    if (m_id && (tx == ty)){
      csrValM[ idx ]    =  d_one - sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_one - sA[threadIdx.y * slda + threadIdx.x];
    }else{
      csrValM[ idx ]    =  d_oneopp*sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_oneopp*sA[threadIdx.y * slda + threadIdx.x];
    }
    idcol = batchid+offset;   
    csrColIndM[ idx ]    = idcol*n + threadIdx.x;
  }
  __syncthreads();

    //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, 0, 0, lddv, sA, slda); // sA = V1.T 
  load_2d_shared(d_cublas_trans_t, dT, 0, 0, lddt, sB, sldb); // sB = T.T
  __syncthreads();
    
  //sC = sB*sA = T.T@V1.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();

  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); // sA = V1
  __syncthreads();

  //sB = sA*sC = V1*T.T*V1.T
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();

  //BLOCCO 11
  if(0){
    device_scrivi_blocco_csr(n, sB, sldb,
			     //csrValM, csrColIndM,
			     csrValMptr[it], csrColIndMptr[it], 
			     batchid, 0, 1,
			     batchid,  offset);
  }else{
    idx = (batchid*n + ty)*2*n + 0*n + tx; //  csrRowPtrM[block_row_idx*n + threadIdx.x] + block*n + threadIdx.y;
    m_id = 1;
    if (m_id && (tx == ty)){
      csrValM[ idx ]    =  d_one - sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_one - sA[threadIdx.y * slda + threadIdx.x];
    }else{
      csrValM[ idx ]    =  d_oneopp*sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_oneopp*sA[threadIdx.y * slda + threadIdx.x];
    }
    idcol = batchid;   
    csrColIndM[ idx ]    = idcol*n + threadIdx.x;
  }
  __syncthreads();

  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); // sA = V2
  __syncthreads();
  
  //sB = sA*sC = V2*T.T*V1.T
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();

  //BLOCCO 21
  if(0){
    device_scrivi_blocco_csr(n, sB, sldb,
			     //csrValM, csrColIndM,
			     csrValMptr[it], csrColIndMptr[it], 
			     batchid+offset, 0, 0,
			     batchid,  offset);
  }else{
    idx = ((batchid+offset)*n + ty)*2*n + 0*n + tx; //  csrRowPtrM[block_row_idx*n + threadIdx.x] + block*n + threadIdx.y;
    m_id = 0;
    if (m_id && (tx == ty)){
      csrValM[ idx ]    =  d_one - sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_one - sA[threadIdx.y * slda + threadIdx.x];
    }else{
      csrValM[ idx ]    =  d_oneopp*sB[tx * sldb + ty];
      //csrValMptr[ idx ] =  d_oneopp*sA[threadIdx.y * slda + threadIdx.x];
    }
    idcol = batchid;   
    csrColIndM[ idx ]    = idcol*n + threadIdx.x;
  }
  __syncthreads();
  
   //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, n, 0, lddv, sA, slda); // V2.T 
  load_2d_shared(d_cublas_trans,   dA, n, 0, lddv, sB, sldb); // A2
  if (ty == 0)
    load_1d_shared(b_array[batchid + offset],  1, sx, 1); //b2
  __syncthreads();
    
  //sC = sA*sB = V2.T@A2
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  //sy = sA*sx = V2.T*b2
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, 0, 0, lddv, sA, slda); // V1.T
  load_2d_shared(d_cublas_trans,   dA, 0, 0, lddv, sB, sldb); // A1
  if (ty == 0)
    load_1d_shared(b_array[batchid],  1, sx, 1); //b1
  __syncthreads();
  
  //sC += sA*sB = V1.T*A1 + (V2.T*A2)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_one, sC, sldc);
  __syncthreads();
  //sy += sA*sx = V1.T*b1 + (V2.T*b2)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_one, sy, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dT, 0, 0, lddt, sA, slda); //sA = T.T
  __syncthreads();
  
  //sB = sA*sC = T.T*(V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);
  __syncthreads();
  
  //sx = sA*sy = T.T*(V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sy, 1, d_zero, sx, 1);
  __syncthreads();
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans, dV, 0, 0, lddv, sA, slda); //sA = V1
  __syncthreads();

  //sC = sA*sB = V1*(T.T*V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();

  //sy = sA*sx = V1*(T.T*V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();

  //scrivo i risultati in global memory
  //dA = dA_array[1];  // scrivo il blocco (1,2) in Aout cosi' e' gia' pronto per il sistema triangolare
  //dA[ty * ldda + tx] -= sC[ty * sldc + tx];
  dA[ty * lddv + tx] -= sC[ty * sldc + tx];
  db = b_array[batchid];
  if (ty == 0)
    db[tx] -= sy[tx];

  
  //CARICO IN SHARED:
  __syncthreads();
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V2
  __syncthreads();
  
  //sC = sA*sB = V2*(T.T*V.T*A)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sB, sldb, d_zero, sC, sldc);
  __syncthreads();
  //sy = sA*sx = V2*(T.T*V.T*b)
  gemv2d_batched_smallsq_device(n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();
  
  //scrivo i risultati in global memory
  //dA = dV_array[1] + 0 * lddv + n; // scrivo il blocco (2,2) in Ain perche' poi ne faccio la QR
  dA +=  0 * lddv + n; // scrivo il blocco (2,2) in Ain perche' poi ne faccio la QR 
  dA[ty * lddv + tx] -= sC[ty * sldc + tx];
  if (ty == 0)
    b_array[batchid + offset][tx] -= sy[tx];

  
  return;
}


/******************************************************************************/
__global__ void larfb_step2_last_call_kernel(int n, tipo **V, int ldv, int vi, int vj, tipo **T, int ldt, tipo **b, tipo_int offset)
{

  /* kernel con un blocco 1d di threads!
     aggiorna il RHS nella seconda QR che si effettua nella risoluzione dei sistemi lineari 2x2
   */
 
  int tx = threadIdx.x;
  int batchid = blockIdx.x;
  
  if ((tx >= n) || (batchid>offset)) return;
 
  
  int slda = SLDA(n);
  
  tipo *sA = (tipo*)(zdata);
  tipo *sx = (tipo*)(zdata + n * slda);
  tipo *sy = (tipo*)(zdata + n * slda + n);

  int idx = batchid + offset;
  tipo *dV = V[idx] + vi + vj*ldv;
  tipo *dT = T[idx];
  tipo *db = b[idx];
  
  //carico shared
  for (int i = 0; i<n; i++)
    sA[tx + i*slda] = dV[tx + i*ldv]; // sA = V
  
  sx[tx] = db[tx]; // sx = b
  __syncthreads();

  // sy = sA.T*sx = V^T*b
  gemv1d_batched_smallsq_device(d_cublas_trans_t, n, d_one, sA, slda, sx, 1, d_zero, sy, 1);
  __syncthreads();

  // sx = T.T*sy = T.T*V^T*b
  gemv1d_batched_smallsq_device(d_cublas_trans_t, n, d_one, dT, ldt, sy, 1, d_zero, sx, 1);
  __syncthreads();

  // b = b - sA*sx = b -  V*T.T*V^T*b
  gemv1d_batched_smallsq_device(d_cublas_trans, n, d_oneopp, sA, slda, sx, 1, d_one, db, 1);
  
  return;
}


/******************************************************************************/
__global__ void larfb_calcolaM_step2_last_call_kernel(int n,
						      tipo **dV_array, int lddv, 
						      tipo **dT_array, int lddt,
						      tipo **csrValMptr, tipo** csrColIndMptr, int it,
						      tipo **b, tipo_int batchCount, tipo_int offset)
{

  /* kernel con un blocco 2d di threads!
     aggiorna il RHS nella seconda QR che si effettua nella risoluzione dei sistemi lineari 2x2
  */
 

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int batchid  = blockIdx.x;
   
  if ((tx >= n) || (ty >= n) || (batchid > (batchCount/2))) return;

  int idx = batchid + offset;
  tipo *dV = dV_array[idx];
  tipo *dT = dT_array[idx];
  tipo *db = b[idx];
  tipo *csrValM = csrValMptr[it];
  
  const int slda = SLDA(n); 
  const int sldb = SLDA(n);
  const int sldc = SLDA(n);
  tipo* sA = (tipo*)(zdata);
  tipo* sB = (tipo*)(zdata + slda * n);
  tipo* sC = (tipo*)(zdata + 2 * slda * n);
  tipo* sx = (tipo*)(zdata + 3 * slda * n );

  
  
  //CARICO IN SHARED:
  load_2d_shared(d_cublas_trans_t, dV, n, 0, lddv, sA, slda); //sA = V.T
  //load_2d_shared(d_cublas_trans,   dA, ai, aj, ldda, sB, sldb); //sB = A
  load_2d_shared(d_cublas_trans_t, dT, 0, 0, lddt, sB, sldb); //sA = T.T
  __syncthreads();
  //sC = sB*sA = T.T*V.T
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);

  //CARICO IN SHARED:
  __syncthreads();
  //load_2d_shared(d_cublas_trans_t, dT, ti, tj, lddt, sA, slda); //sA = T.T
  load_2d_shared(d_cublas_trans, dV, n, 0, lddv, sA, slda); //sA = V
  __syncthreads();
  //sB = sA*sC = V*(T.T*V.T)
  device_gemm_batched_smallsq_kernel(n, d_one, sA, slda, sC, sldc, d_zero, sB, sldb);

  
  //CARICO IN SHARED:
  if ((tx < n) && (ty < n)){
    sA[tx * slda + ty] = csrValM[((batchid+offset)*n + ty)*2*n + 0*n + tx];
  }
  __syncthreads();

  
  //sC = sB*sA
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();


  // SCRIVO  in GLOBAL
  if ((tx < n) && (ty < n))
    csrValM[((batchid+offset)*n + ty)*2*n + 0*n + tx ] -= sC[tx * sldc + ty];
  
  //CARICO IN SHARED:
  //load_2d_shared_csr((batchid+offset)*n+n, n, csrValMptr[it], csrColIndMptr[it],  sA, slda);
  if ((tx < n) && (ty < n))
    sA[tx * slda + ty] = csrValM[((batchid+offset)*n + ty)*2*n + 1*n + tx ];
  __syncthreads();

  //sC = sB*sA
  device_gemm_batched_smallsq_kernel(n, d_one, sB, sldb, sA, slda, d_zero, sC, sldc);
  __syncthreads();


  // SCRIVO  in GLOBAL
  if ((tx < n) && (ty < n))
    csrValM[((batchid+offset)*n + ty)*2*n + 1*n + tx ] -= sC[tx * sldc + ty];
  
  if (ty == 0){
    sx[tx] = db[tx]; // sx = b
    __syncthreads();
    // b = b - sB*sx = b -  (V*T.T*V^T)*b
    gemv1d_batched_smallsq_device(d_cublas_trans, n, d_oneopp, sB, sldb, sx, 1, d_one, db, 1);
  }
  
  return;
}
