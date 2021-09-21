/* trsv_device.cu -- modified by MAGMA http://icl.cs.utk.edu/magma/
   M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
*/

extern __shared__ tipo shared_data[];

static __device__ void
dtrsv_backwards_tri_device(int unitdiag, cublasOperation_t transA,
			   int n,
			   const tipo * __restrict__ A, int lda,
			   tipo       * __restrict__ b, int incb,
			   tipo *sx)

{

  /* risolve Ax = b/A^Tx=b con backward substitution, 
     A e' triangolare superiore
     unitdiag = 1 => A ha solo 1 sulla diagonale
     
     assume sx is in shared memory
  */
  
  int tx = threadIdx.x;
  tipo a;
  
  for (int step=0; step < n; step++) // risolve sx[step]
    {
      if (tx < n)
        {
	  if (transA == d_cublas_trans)
            {
	      a = A[ (n-1) + (n-1) * lda - tx - step * lda]; // rowwise access data in a coalesced way; accesso alla colonna step-esima A[n-1-tx, n-1-step] 
            }
	  else if (transA == d_cublas_trans_t)
            {
	      a = A[ (n-1) + (n-1) * lda - tx * lda  - step]; // columwise access data, not in a coalesced way A[n-1-step, n-1-tx]
            }
	  
	  if (tx == step) // calcolo x[step]
            {
	      if (unitdiag == 1)
                {
		  sx[n-1-tx] = (b[n-1-tx] - sx[n-1-tx]);
                }
	      else
                {
		  sx[n-1-tx] = (b[n-1-tx] - sx[n-1-tx])/a;
                }

            }
        }
      __syncthreads(); // there should be a sych here but can be avoided if BLOCK_SIZE =32

      if (tx < n) // aggiorno il RHS per le righe restanti
        {
	  if (tx > step)
            {
	      sx[n-1-tx] += a * sx[n-1-step];
            }
        }
    }
}

__global__ void dtrsv_last_call_kernel(int n,
				       tipo **A, int lda,
				       tipo **b, tipo **x, tipo_int offset)
{  
  /* risolve Ax = b, dove A e' triangolare superiore ottenuta all'ultimo step di parasof
     vuole un unico blocco di thread unidimensionale */

  int tx = threadIdx.x;
  int batchid = blockIdx.x;
  if (batchid > offset)
    return;
  int idx = batchid + offset;
  
  tipo *dx = x[idx];
  tipo *db = b[idx];
  tipo *dA = A[idx] + 0*lda + n;
  tipo *sx = (tipo*)shared_data;


  // if (tx == 0){
  //   printf("batchid %d \n %f %f %f %f \n 0 %f %f %f \n 0 0 %f %f \n 0 0 0 %f \n rhs = %f %f %f %f \n", batchid,
  // 	   *(A[batchid] + 0*lda + 0), *(A[batchid] + 1*lda + 0), *(A[idx] + 0*lda + 0),  *(A[idx] + 1*lda + 0),
  // 	   *(A[batchid] + 1*lda + 1), *(A[idx] + 0*lda + 1), *(A[idx] + 1*lda + 1),
  // 	   *(A[idx] + 0*lda + 2),  *(A[idx] + 1*lda + 2),
  // 	   *(A[idx] + 1*lda + 3),
  // 	   b[batchid][0], b[batchid][1], b[idx][0], b[idx][1] );
  //    }
  // __syncthreads();
  
  if (tx < n)
    {
      sx[tx] = d_zero;
    }
  __syncthreads();

  // risolvo A22 x2 = b2
  dtrsv_backwards_tri_device(0, d_cublas_trans, n, dA, lda, db, 1, sx);
  __syncthreads();
  
  if (tx < n)
    {
      dx[tx] = sx[tx]; // write to x in reverse order
    }
  __syncthreads();
  
  dA = A[idx] + 0*lda + 0;
  gemv1d_batched_smallsq_device(d_cublas_trans, n, d_one, dA, lda, sx, 1, d_zero, db, 1);

  if (tx < n)
    {
      sx[tx] = db[tx];
    }
  __syncthreads();
  
  dA = A[batchid] + 0*lda + 0;
  dx = x[batchid];
  db = b[batchid];
  
  dtrsv_backwards_tri_device(0, d_cublas_trans, n, dA, lda, db, 1, sx);
  __syncthreads();
  
  if (tx < n)
    {
      dx[tx] = sx[tx]; // write to x in reverse order
    }
  __syncthreads();

  
}
__global__ void sof_dtrsv_batched_kernel(int n,
					 tipo **A, int lda,
					 tipo **R, int ldr,
					 tipo **b, tipo **x,
					 int nb_slices, int slice_size, tipo_int batchCount)
{  
  /* risolve Ax = b, dove A e' triangolare superiore ottenuta all'ultimo step di parasof
     vuole un unico blocco di thread unidimensionale */

  int tx = threadIdx.x;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,nb_slices -1
  int sliceId = bx * blockDim.z + tz; // = bx per ogni blocco

  tipo *dx, *db, *dA;
  tipo *sx = (tipo*)shared_data;
  tipo *sy = (tipo*)shared_data + blockDim.z*n;
  sx += tz * n;
  sy += tz * n;

  int idx, batchIdR, batchIdA;
  
  if (sliceId >= nb_slices) return;

  for (int offset = slice_size - 1; offset > 0; offset--){ 
    idx = sliceId * slice_size + (offset-1); //risolve equazione idx per trovare x[idx+1]
    batchIdR = sliceId * (slice_size - 1) + (offset-1);
    batchIdA = idx + 1;

    if ( (batchIdA+1) >= batchCount ) 
      return;
    
    dA = A[batchIdA]; // W
    dx = x[batchIdA + 1];
    if (tx < n)
      {
	sx[tx] = dx[tx];
      }
    __syncthreads();
    
    // sy = sA*sx = W*x_{i+1}
    gemv1d_batched_smallsq_device(d_cublas_trans, n, d_one, dA, lda, sx, 1, d_zero, sy, 1);

    // __syncthreads();
    // if (tx == 0)
    //   printf("sliceId %d, offset %d, %f %f  \n", sliceId, offset, sy[0], sy[1]  );
    // __syncthreads();
    
    dA = A[batchIdA-1] + 0*lda + n; // V
    dx = x[sliceId*slice_size];
    if (tx < n)
      {
	sx[tx] = dx[tx];
      }
    __syncthreads();
    
    // sy += sA*sx = V*x{0} + W*x_{i+1}
    gemv1d_batched_smallsq_device(d_cublas_trans, n, d_one, dA, lda, sx, 1, d_one, sy, 1);

    // __syncthreads();
    // if (tx == 0)
    //   printf("sliceId %d, offset %d, %f %f  \n", sliceId, offset, sy[0], sy[1]  );
    __syncthreads();
    
    dA = R[batchIdR];
    db = b[idx];
    
    dtrsv_backwards_tri_device(0, d_cublas_trans, n, dA, ldr, db, 1, sy);
    __syncthreads();

    dx = x[batchIdA];
    
    if (tx < n)
      {
	dx[tx] = sy[tx]; // write to x in reverse order
      }
    
    __syncthreads();
    
  }
}

__global__ void sof_dtrsv_batched_kernel_SOLV(int n,
					      tipo **A, int lda,
					      tipo **R, int ldr,
					      tipo **b, tipo **x,
					      int nb_slices, int slice_size, tipo_int batchCount)
{  
  /* risolve Ax = b, dove A e' triangolare superiore ottenuta all'ultimo step di parasof
     vuole un unico blocco di thread unidimensionale */

  int tx = threadIdx.x;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,nb_slices -1
  int sliceId = bx * blockDim.z + tz; // = bx per ogni blocco

  tipo *dx, *db, *dA;
  tipo *sx = (tipo*)shared_data;
  tipo *sy = (tipo*)shared_data + blockDim.z*n;
  sx += tz * n;
  sy += tz * n;

  int idx, batchIdR, batchIdA;
  int idx0 = sliceId * slice_size;
  
  if (sliceId >= nb_slices) return;

  for (int offset = slice_size - 1; offset > 0; offset--){
    
    idx = idx0 + (offset-1); 
    batchIdR = idx0 - sliceId + (offset-1);
    batchIdA = idx + 1; // idx0 + (offset-1) +1 =  idx0 + offset = idx0 + slice_size - 1

    if ( (batchIdA+1) >= batchCount ) 
      return;
    
    dA = A[batchIdA]; // W
    dx = x[batchIdA + 1];
    if (tx < n)
      {
	sx[tx] = dx[tx];
      }
    __syncthreads();

    // sy = sA*sx = W*x_{i+1}
    gemv1d_batched_smallsq_device(d_cublas_trans, n, d_one, dA, lda, sx, 1, d_zero, sy, 1);

    // __syncthreads();
    // if (tx == 0)
    //   printf("sliceId %d, offset %d, %f %f  \n", sliceId, offset, sy[0], sy[1]  );
    // __syncthreads();
    // if (tx == 0){
    //   printf("batchid %d \n %f %f %f %f \n 0 %f %f %f \n 0 0 %f %f \n 0 0 0 %f \n rhs = %f %f %f %f \n sol =  %f %f %f %f  \n", batchid,
    // 	     *(A[batchid] + 0*lda + 0), *(A[batchid] + 1*lda + 0), *(A[idx] + 0*lda + 0),  *(A[idx] + 1*lda + 0),
    // 	     *(A[batchid] + 1*lda + 1), *(A[idx] + 0*lda + 1), *(A[idx] + 1*lda + 1),
    // 	     *(A[idx] + 0*lda + 2),  *(A[idx] + 1*lda + 2),
    // 	     *(A[idx] + 1*lda + 3),
    // 	     b[batchid][0], b[batchid][1], b[idx][0], b[idx][1], x[batchid][0], x[batchid][1], x[idx][0], x[idx][1] );
    // }
    // __syncthreads();
    
    if (offset == 1)
      dA = A[idx0 + slice_size -1] + 0*lda + n; // V_0 e' stata scritta nell'ultimo blocco della slice
    else
      dA = A[batchIdA-1] + 0*lda + n; // V
    
    dx = x[idx0];
    if (tx < n)
      {
	sx[tx] = dx[tx];
      }
    __syncthreads();
    
    // sy += sA*sx = V*x{0} + W*x_{i+1}
    gemv1d_batched_smallsq_device(d_cublas_trans, n, d_one, dA, lda, sx, 1, d_one, sy, 1);

    // __syncthreads();
    // if (tx == 0)
    //   printf("sliceId %d, offset %d, %f %f  \n", sliceId, offset, sy[0], sy[1]  );
    __syncthreads();
    
    dA = R[batchIdR];
    db = b[idx];
    
    dtrsv_backwards_tri_device(0, d_cublas_trans, n, dA, ldr, db, 1, sy);
    __syncthreads();

    dx = x[batchIdA];
    
    if (tx < n)
      {
	dx[tx] = sy[tx]; // write to x in reverse order
      }
    
    __syncthreads();
    
  }
}
