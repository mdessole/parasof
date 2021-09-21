/* geqr2.cu -- modified by MAGMA http://icl.cs.utk.edu/magma/
   M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
*/

#define max_shared_bsiz 32
#define use_gemm_larft

extern __shared__ tipo shared_data[];

static __device__ void
larfg_device(int n,
	     tipo* dalpha, tipo* dx, int incx, tipo* dr, int ldr,
	     tipo* dtau,  tipo* swork, tipo* sscale, tipo* scale,
	     int calcola_R)
{
  /*
    costruisce il vettore di householder v t.c. 
    Hx = (I - tau*vv^T)w = (beta 0)^T, HH^T = I, 
    v = (1 dv)^T, x = (alpha, dx)^T
    ***
    Input:

    n = lunghezza del vettore x = (alpha, dx)^T
    alpha = prima entrata del vettore v
    dx = le ultime (n-1) entrate di v
    incx = offset tra due elementi consecutivi di x (se dx e' un vettore colonna, allora incx = 1)

    Output:
    tau = vedi espressione sopra
    dx = viene sovrascritto con dv

    NUMERO DI THREAD NECESSARI = n

    *******************************************************************************************************************
    IMPORTANTE
    *******************************************************************************************************************
    LAPACK: alpha = x1, beta = norm(x), vettore H. v = (beta - x1, x2, ..., xn).T*1/(beta - x1), tau = (beta - alpha)/beta = (beta - x1)/beta,
    Matrice H: H = I - tau v v.T
    Hx = (beta,0,...,0).T
    *******************************************************************************************************************
  */
  const int tx = threadIdx.x; 
  tipo tmp;
    
  // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow    

  if ( tx == 0 ) {
    tmp = *dalpha;
    swork[tx] = fabs(tmp);
  }
  else {
    swork[tx] = 0;
  }
  if (tx < BLOCK_SIZE)
    {
      for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
	tmp = dx[j*incx];
	swork[tx] = max( swork[tx], fabs(tmp) );
      }
    }

  // swork[0] = max(swork)
  magma_max_reduce(BLOCK_SIZE, tx, swork );

  if ( tx == 0 )
    *sscale = swork[0];
  // __syncthreads();
    
  // sum norm^2 of dx/sscale
  // dx has length n-1
  if (tx < BLOCK_SIZE) swork[tx] = 0;
  if ( *sscale > 0 ) {
    if (tx < BLOCK_SIZE)
      {
	for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
	  tmp = dx[j*incx] / *sscale;
	  swork[tx] += tmp*tmp;
	}
      }
    magma_sum_reduce(BLOCK_SIZE, tx, swork );
    //swork[0] = sum(swork) = norm(dx/sscale)**2
  }
    
  if ( tx == 0 ) {
    tipo alpha = *dalpha;

    if ( swork[0] == 0 && alpha == 0) { //( swork[0] == 0 && imag(alpha) == 0 )
      // H = I
      *dtau = 0;
    }
    else {
      // beta = norm( [dalpha, dx] )
      tipo beta;
      tmp  = alpha / *sscale;
      beta = *sscale * sqrt( tmp*tmp + swork[0] );
      beta = -copysign( beta, alpha );
      // todo: deal with badly scaled vectors (see lapack's larfg)
      *dtau   = (beta - alpha) / beta;
      *dalpha = 1; //MAGMA_S_MAKE( beta, 0 );
      *scale = 1 / (alpha - beta);

      if (calcola_R)
	dr[tx] = beta; //thread 0 writes on the diagonal
      //printf("%f\n", beta);
    }

  }
  else if ((tx < n) && (tx < ldr) && calcola_R)
    dr[tx] = d_zero;
    
  // scale x (if norm was not 0)
  // __syncthreads();
  if ( swork[0] != 0 ) {
    if (tx < BLOCK_SIZE)
      {
	for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
	  dx[j*incx] *= *scale;
	}
      }
  }
  
  return;
}

/******************************************************************************/
static __device__
void larfx_device(int m, int n,  tipo *v, tipo *tau,
		  tipo *dc, int ldc,
		  tipo *dr, int ldr,
		  tipo* sum,
		  int calcola_R)
{
  // (m, n, dv, dtau, dA, lda, sum)
  // NUMERO DI THREAD NECESSARI = m
  if (n <= 0) return;
  if (*tau == 0.0 )  return; // check singularity

  const int tx = threadIdx.x;
  tipo lsum, z__1;
  int k, j;
  
  for (k=0; k < n; k++)
    {
      /* perform  w := v' * C  */
      if (tx < BLOCK_SIZE)
        {
	  if (tx == 0)
	    lsum = dc[0+ldc*k]; //since V[0] should be 1
	  else
	    lsum = 0.0;
	  // finto for se m <= BLOCKSIZE
	  for (j = tx+1; j < m; j += BLOCK_SIZE) {
	    lsum += v[j] * dc[j+ldc*k]; //lsum += v[j]*dc[j,k]
	  }

	  sum[tx] = lsum;
        }

      // sum[0] = sum(sum)	
      magma_sum_reduce(BLOCK_SIZE, tx, sum );

	
      z__1 = - (*tau) * sum[0];
      /*  C := C - v * w  */
      if (tx < BLOCK_SIZE)
	//dc[(tx+1)+ldc*k] += z__1 * v[tx+1]; // dc[j,k];
        {
	  // finto for se m <= BLOCKSIZE
	  for (j = tx+1; j < m; j += BLOCK_SIZE)
	    dc[j+ldc*k] += z__1 * v[j]; // dc[j,k]
        }
      if (tx == 0) {
	if (calcola_R)
	  dr[0+ldr*k] = dc[0+ldc*k]+z__1; // Calcola R
	dc[0+ldc*k] = 0.0;
      }
      
      __syncthreads();
    }
  
  return;
}


__device__
void scambia_blocco( int m, int n, tipo* dA, int lda)
{//il primo blocco deve convertire [[B_a],[S_0]] in [[S_0],[B_a]]
  // int tx = threadIdx.x;
  // if (tx >= m)
  //   return;

  // tipo val;
  
  // for (int s=0; s<n; s++){
  //   val = dA[tx + s*lda];
  //   __syncthreads();
  //   if (tx<n)
  //     dA[(tx+n) + s*lda] = val;
  //   else
  //     dA[(tx-n) + s*lda] = val;
  // }

  int tx = threadIdx.x;
  if (tx >= n)
    return;
  
  const int bx = blockIdx.z; // in questo caso  0,...,batchCount -1
  int batchid = bx * blockDim.z;
  
  tipo tmp;
  int s;
  
  for (s=0; s<n; s++){
    tmp = dA[tx + s*lda];
    dA[tx + s*lda] = dA[(tx+n) + s*lda];
    dA[(tx+n) + s*lda] = tmp;
  }
  
  return;
}


__device__ void leggi_2d_shared_prova(int nrows, int ncols, tipo* dA, int ai, int aj, int ldda, tipo* sA, int slda){
  tipo* dA_tmp = dA + aj * ldda + ai; // dA = dA_array[batchid][ai:,aj:]
  int tx = threadIdx.x, i;
  
  if (tx >= nrows)
    return;

  for (i=0;i<ncols;i++)
    sA[i * slda + tx] = dA_tmp[tx + i * ldda];

  return;
}

__device__ void scrivi_2d_shared_prova(int nrows, int ncols, tipo* dA, int ai, int aj, int ldda, tipo* sA, int slda){
  tipo* dA_tmp = dA + aj * ldda + ai; // dA = dA_array[batchid][ai:,aj:]
  int tx = threadIdx.x, i;
  
  if (tx >= nrows)
    return;

  for (i=0;i<ncols;i++)
    dA_tmp[tx + i * ldda] = sA[i * slda + tx];

  return;
}


/******************************************************************************/
static __device__
void geqr2_device( int m, int n,
		   tipo *dA, int lda,
		   tipo *dR, int ldr,
		   tipo *dtau,
		   tipo *dv,
		   tipo *dw,
		   tipo *sum,
		   tipo *swork,
		   tipo *scale,
		   tipo *sscale,
		   int calcola_R)
{ 
  //lapack dlarfg, compute the norm, scale and generate the householder vector
  // larfg_device(int n,
  // 	     tipo* dalpha, tipo* dx, int incx, tipo* dr, int ldr,
  // 	     tipo* dtau,  tipo* swork, tipo* sscale, tipo* scale,
  // 	     int calcola_R)
  
  larfg_device(m, dv, &(dv[1]), 1, dw, 1, dtau, swork, sscale, scale, calcola_R);
  __syncthreads(); 
  
  
  //update the trailing matix with the householder  
  // larfx_device(int m, int n,  tipo *v, tipo *tau,
  // 		  tipo *dc, int ldc,
  // 		  tipo *dr, int ldr,
  // 		  tipo* sum,
  // 		  int calcola_R)
  larfx_device(m, n, dv, dtau, dA, lda, dR, ldr, sum, calcola_R);
  __syncthreads();
}


/******************************************************************************/
__global__
//__launch_bounds__(32*2, 16)
void geqr2_kernel_batched(int m, int n, tipo** dA_array, int lda, int ai, int aj,
			  tipo **dR_array, int ldr, int ri, int rj,
			  tipo **dtau_array, int calcola_R, int inverti_blocchi,
			  tipo_int batchCount, tipo_int offset)
{
  //const int tx = threadIdx.x;
  //const int ty = threadIdx.y;
  const int tz = threadIdx.z; // 0,...,ntcols-1, in questo caso 0 per ogni thread
  const int bx = blockIdx.z; // in questo caso  0,...,batchCount -1
  
  int batchid = bx * blockDim.z + tz; // = bx per ogni blocco
  //int batchid = blockIdx.z*QRperBlock;
  if (batchid >= batchCount)
    return;
  
  tipo* sum   = (tipo*)(zdata);
  tipo* swork = (tipo*)(zdata + blockDim.z*BLOCK_SIZE);
  tipo* scale  = (tipo*)(zdata + blockDim.z*BLOCK_SIZE*2);
  tipo* sscale = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+1));
  sum   += tz * BLOCK_SIZE;
  swork += tz * BLOCK_SIZE;
  scale += tz * 1;
  sscale += tz * 1;

  tipo* dA   = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+2));// = dA_array[batchid] + aj * lda + ai; 
  dA  += tz * lda*n;
  tipo* dtau = dtau_array[batchid];
  tipo* dR   = dR_array[batchid] + rj * ldr + ri;
  int s;

  // if ((threadIdx.x == 0) && (threadIdx.y == 0))
  //   printf("batchid A = %d\n", batchid);
  
  leggi_2d_shared_prova(m, n, dA_array[batchid], ai, aj, lda, dA, lda);
  __syncthreads();
  
  // il primo blocco rettangolare e' salvato con i blocchi invertiti, prima di fare la QR li rimetto com'erano
  // nell'ultimo passo di ricorsione chiamo questa funzione su una matrice quadrata che non ha bisogno di questo lavoro, percio' la condizione (m != n )
  //if ((m != n ) && (batchid < offset)){
  //  scambia_blocco( m, n, dA, lda);
  //}
  if ((inverti_blocchi) && (batchid < offset)){
    scambia_blocco( m, n, dA, lda);
    __syncthreads();
  }

    
  // threads necessari = max(m,n)
  for (s=0; s < min(m,n); s++)
    {
      geqr2_device( m-s, n-(s+1),
		    &(dA[s+(s+1)*lda]), lda,
		    &(dR[s+(s+1)*lda]), ldr,
		    dtau+s, 
		    &(dA[s+s*lda]),
		    &(dR[s+s*lda]),
		    sum,
		    swork,
		    &(scale[0]),
		    &(sscale[0]),
		    calcola_R);
    }

  scrivi_2d_shared_prova(m, n, dA_array[batchid], ai, aj, lda, dA, lda);
  __syncthreads();
  return;
}

__global__ 
void sof_geqr2_kernel_batched( int m, int n,
			       tipo** dA_array, int lda, int ai, int aj,
			       tipo** dV_array, int ldv, int vi, int vj,
			       tipo** dR_array, int ldr, int ri, int rj,
			       tipo **dtau_array,
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
  int batchId_R = sliceId * (slice_size - 1) + (offset-1);

  if ((sliceId >= nb_slices) || (batchId_A >= batchCount))
    return;
  
  tipo* sum   = (tipo*)(zdata);
  tipo* swork = (tipo*)(zdata + blockDim.z*BLOCK_SIZE);
  tipo* scale  = (tipo*)(zdata + blockDim.z*BLOCK_SIZE*2);
  tipo* sscale = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+1));
  sum   += tz * BLOCK_SIZE;
  swork += tz * BLOCK_SIZE;
  scale += tz * 1;
  sscale += tz * 1;
  tipo* dA   = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+2));// = dA_array[batchid] + aj * lda + ai; //
  dA  += tz * lda*n;
  
  tipo* dtau = dtau_array[sliceId];
  tipo* dR   = dR_array[batchId_R] + rj * ldr + ri;
  int s;
  
  leggi_2d_shared_prova(m, n, dA_array[batchId_A], ai, aj, lda, dA, lda);
  __syncthreads();
    
  // threads necessari = max(m,n)
  for (s=0; s < min(m,n); s++)
    {
      geqr2_device( m-s, n-(s+1),
		    &(dA[s+(s+1)*lda]), lda,
		    &(dR[s+(s+1)*ldr]), ldr,
		    dtau+s, 
		    &(dA[s+s*lda]),
		    &(dR[s+s*ldr]),
		    sum,
		    swork,
		    &(scale[0]),
		    &(sscale[0]),
		    int(1) );
    }
  
  scrivi_2d_shared_prova(m, n, dV_array[sliceId], ai, aj, lda, dA, lda);
  __syncthreads();
  return;
}

__global__ 
void sof_geqr2_kernel_batched_FACT( int m, int n,
				    tipo** dA_array, int lda, int ai, int aj,
				    tipo** dV_array, int ldv, int vi, int vj,
				    tipo** dR_array, int ldr, int ri, int rj,
				    tipo **dtau_array,
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
  int batchId_R = sliceId * (slice_size - 1) + (offset-1);

  if ((sliceId >= nb_slices) || (batchId_A >= batchCount))
    return;
  
  tipo* sum   = (tipo*)(zdata);
  tipo* swork = (tipo*)(zdata + blockDim.z*BLOCK_SIZE);
  tipo* scale  = (tipo*)(zdata + blockDim.z*BLOCK_SIZE*2);
  tipo* sscale = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+1));
  sum   += tz * BLOCK_SIZE;
  swork += tz * BLOCK_SIZE;
  scale += tz * 1;
  sscale += tz * 1;
  tipo* dA   = (tipo*)(zdata + blockDim.z*(BLOCK_SIZE*2+2));// = dA_array[batchid] + aj * lda + ai; //
  dA  += tz * lda*n;
  
  tipo* dtau = dtau_array[sliceId];
  tipo* dR   = dR_array[batchId_R] + rj * ldr + ri;
  int s;
  
  leggi_2d_shared_prova(m, n, dA_array[batchId_A], ai, aj, lda, dA, lda);
  __syncthreads();
    
  // threads necessari = max(m,n)
  for (s=0; s < min(m,n); s++)
    {
      geqr2_device( m-s, n-(s+1),
		    &(dA[s+(s+1)*lda]), lda,
		    &(dR[s+(s+1)*ldr]), ldr,
		    dtau+s, 
		    &(dA[s+s*lda]),
		    &(dR[s+s*ldr]),
		    sum,
		    swork,
		    &(scale[0]),
		    &(sscale[0]),
		    int(1) );
    }
  
  scrivi_2d_shared_prova(m, n, dV_array[batchId_A], ai, aj, lda, dA, lda);
  __syncthreads();
  return;
}

extern "C" void
magma_dgeqr2_batched(int m, int n, 
                     tipo **dA_array, int ldda, int ai, int aj,
		     tipo **dR_array, int lddr, int ri, int rj,
                     tipo **dtau_array,
		     int calcola_R, int inverti_blocchi,
                     tipo_int batchCount)
{
  /* Check arguments */
  int arginfo = 0;
  if (m < 0)
    arginfo = -1;
  else if (n < 0)
    arginfo = -2;
  else if (ldda < max(1,m))
    arginfo = -4;
  if (arginfo != 0) {
    printf("ERRORE\n");
    return;
  }

  dim3 blocks(1, 1, batchCount);
  dim3 threads(BLOCK_SIZE); //(BLOCK_SIZE)
  

  
  // if (sizeof(tipo)*(m*k) <= 42000 /*sizeof(tipo) * 128 * k*/) // there are some static shared memory besides of dynamic ones
  // {
  //     //load panel in shared memory and factorize it and copy back to gloabl memory
  //     //intend for small panel to avoid overfill of shared memory.
  //     //this kernel is composed of device routine and thus clean
  //     dgeqr2_sm_kernel_batched<<< blocks, threads, sizeof(tipo)*(m*k), queue->cuda_stream() >>>
  //                                   (m, k, dA_array, ldda, dtau_array);
  // }
  // else
  // {
  //     //load one column vector in shared memory and householder it and used it to update trailing matrix which is global memory
  //     // one vector is normally smaller than  48K shared memory
  //     if (sizeof(tipo)*(m) < 42000)
  //         dgeqr2_column_sm_kernel_batched<<< blocks, threads, sizeof(tipo)*(m), queue->cuda_stream() >>>
  //                                   (m, k, dA_array, ldda, dtau_array);
  //     else
  //         //not use dynamic shared memory at all
  //         dgeqr2_kernel_batched<<< blocks, threads, 0, queue->cuda_stream() >>>
  //                                   (m, k, dA_array, ldda, dtau_array);
  // }

  
  // Questa implementazione prevede che un blocco processi una matrice del batch
  geqr2_kernel_batched<<< blocks, threads>>> (m, n, dA_array, ldda, ai, aj, dR_array, lddr, ri, rj, dtau_array, calcola_R, inverti_blocchi, batchCount, int(1));

  
  return;
}


/******************************************************************************/
__global__ void zdisplace_pointers_kernel(tipo **output_array,
					  tipo **input_array, int lda,
					  int row, int column)
{
  tipo *inpt = input_array[blockIdx.x];
  output_array[blockIdx.x] = &inpt[row + column * lda];
  return;
}

/******************************************************************************/
__global__ void sx_displace_pointers_kernel(tipo **output_array,
					    tipo **input_array, int lda,
					    int row, int column, tipo_int batchCount)
{
  int sx_blockIdx;
  if (blockIdx.x == 0)
    sx_blockIdx = batchCount -1;
  else
    sx_blockIdx = blockIdx.x - 1;
    
  tipo *inpt = input_array[sx_blockIdx];
  output_array[blockIdx.x] = &inpt[row + column * lda];
  return;
}

/******************************************************************************/
__global__ void dx_displace_pointers_kernel(tipo **output_array,
					    tipo **input_array, int lda,
					    int row, int column, tipo_int batchCount)
{
  int dx_blockIdx;
  if (blockIdx.x == batchCount -1)
    dx_blockIdx = 0;
  else
    dx_blockIdx = blockIdx.x + 1;
    
  tipo *inpt = input_array[dx_blockIdx];
  output_array[blockIdx.x] = &inpt[row + column * lda];
  return;
}

extern "C"
void magma_ddisplace_pointers(tipo **output_array,
               tipo **input_array, int lda,
               int row, int column, 
               tipo_int batchCount)
{
  /***************************************************************************//**
    Purpose
    -------

    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    
    Arguments
    ----------

    @param[out]
    output_array    Array of pointers, dimension (batchCount).
             Each pointer points to the new displacement of array A in input_array on the GPU
   
    @param[in]
    input_array     Array of pointers, dimension (batchCount).
             Each is a TIPO PRECISION array A of DIMENSION ( lda, column ) on the GPU

    @param[in]
    lda    INTEGER
            LDA specifies the leading dimension of A.

    @param[in]
    row       INTEGER
            On entry, row specifies the number of rows of the matrix A.

    @param[in]
    column       INTEGER
            On entry, column specifies the number of columns of the matrix A

    @param[in]
    batch_offset  INTEGER
                The starting pointer of each matrix A in input arrray

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
    zdisplace_pointers_kernel<<< batchCount, 1>>>(output_array, input_array, lda, row, column);
    
    return;
}

/******************************************************************************/
static  __device__ void 
dlarft_dtrmv_sm32x32_device(
			    int n, int k, tipo *tau,
			    tipo *Tin, int ldtin,  tipo *Tout, int ldtout )
{
    int tx = threadIdx.x; 
    tipo *sdata = (tipo*)shared_data;
    tipo res;

    // This routine applies a sequence of trmv to update k column of the triangular
    // T starting at n-k to n where T is of size n by n and where the first n-k 
    // columns of T are supposed updated previously.
    // So the routine load all of T nxn to the shared memory 
    // and apply the sequence of trmv.
    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate 
    // one element of the column of T then move to the next column

    // bastano m threads

    // // read T into shared
    // for (int s=0; s < n-k; s++)
    // {
    //     sdata[tx + s*n] = Tin[tx + s * ldtin];
    // }
    // // __syncthreads(); 
    
#if defined(use_gemm_larft)
    for (int s=n-k; s < n; s++)
    {
        if (tx == s)
	  sdata[tx + s*n] = tau[s]; //sdata[s,s] = tau[s]
        else
	  sdata[tx + s*n] = -tau[s] * Tin[tx + s * ldtin]; //sdata[tx,s] = -tau[s]*T[tx,s]
    }
#else
    for (int s=n-k; s < n; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }
#endif

    // perform trmv
    for (int i=n-k; i < n; i++)
    {
        // __syncthreads();  
        res = d_zero;
        if (tx < i)
        {
            for (int j=tx; j < i; j++)
            {
                res += sdata[tx + j * n] * sdata[j+ i * n];      
            }
        // }       
        // // __syncthreads();  
        // if (tx < i)
        // {
            sdata[tx + i * n] = res;
        }
    } 

    // __syncthreads();  
    // write back the updated block of k column of T
    for (int s=n-k; s < n; s++)
    {
        Tout[tx + s * ldtout] = sdata[tx + s*n];
    }
    // __syncthreads();
    return;
}

/******************************************************************************/
__global__ void 
dlarft_dtrmv_sm32x32_kernel_batched(
    int n, int k, tipo **tau_array,
    tipo **Tin_array, int ldtin,  tipo **Tout_array, int ldtout )
{
    int batchId = blockIdx.z;
    dlarft_dtrmv_sm32x32_device( n, k, tau_array[batchId], Tin_array[batchId], ldtin, Tout_array[batchId], ldtout);
}

/******************************************************************************/
extern "C"
void magmablas_dlarft_dtrmv_sm32x32_batched(
    int m, int n, 
    tipo **tau_array, 
    tipo **Tin_array, int ldtin, 
    tipo **Tout_array, int ldtout,
    tipo_int batchCount)
{
    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(tipo)*(m*m);
    dlarft_dtrmv_sm32x32_kernel_batched
      <<< grid, threads, shmem >>>
        (m, n,  tau_array, Tin_array, ldtin, Tout_array, ldtout);
}


/******************************************************************************/
static  __device__ void 
dlarft_dtrmv_sm32x32_device_mio(int n, tipo *tau, tipo *T, int ldt)
{
    int tx = threadIdx.x;
    int tz = threadIdx.z; 
    tipo *sdata = (tipo*)shared_data;
    sdata += tz * n * n;
    tipo res;

    // This routine applies a sequence of trmv to update k column of the triangular
    // T starting at n-k to n where T is of size n by n and where the first n-k 
    // columns of T are supposed updated previously.
    // So the routine load all of T nxn to the shared memory 
    // and apply the sequence of trmv.
    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate 
    // one element of the column of T then move to the next column

    // NB: Bastano n threads.
    // Pongo k=n penche' voglio che tutte le n colonne di T vengano aggiornate.
    int k = n;

    if (tx >= n)
      return;
    
    // // read T into shared
    // for (int s=0; s < n-k; s++)
    // {
    //   sdata[tx + s*n] = T[tx + s * ldt]; //sdata[tx,s] = T[tx,s]
    // }
    // // __syncthreads(); 
    
#if defined(use_gemm_larft)
    for (int s=n-k; s < n; s++)
      {
        if (tx == s)
	  sdata[tx + s*n] = tau[s]; //sdata[s,s] = tau[s]
        else
	  sdata[tx + s*n] = -tau[s] * T[tx + s * ldt]; //sdata[tx,s] = -tau[s]*T[tx,s]
      }
#else
    for (int s=n-k; s < n; s++)
      {
        sdata[tx + s*n] = T[tx + s * ldt];
      }
#endif
    
    // perform trmv
    for (int i=n-k; i < n; i++)
      {
        // __syncthreads();  
        res = d_zero;
        if (tx < i)
	  {
            for (int j=tx; j < i; j++)
            {
                res += sdata[tx + j * n] * sdata[j+ i * n];      // sdata[tx,j]*sdata[j,i]
            }
        }       
        // __syncthreads();  
        if (tx < i)
        {
	  sdata[tx + i * n] = res; // sdata[tx,i] = res
        }
    } 

    // __syncthreads();  
    // write back the updated block of k column of T
    for (int s=n-k; s < n; s++)
    {
      T[tx + s * ldt] = sdata[tx + s*n]*(tx<=s); //T[tx,s] = sdata[tx,s] if tx<=s, 0.0 otherw. (in order to write only upper triang. part)
    }
    // __syncthreads();
    return;
}

/******************************************************************************/
__global__ void 
dlarft_dtrmv_sm32x32_kernel_batched_mio(int n, tipo **tau_array, tipo **T_array, int ldt, tipo_int batchCount )
{
  tipo_int batchid = blockIdx.z;
   
  if (batchid>batchCount)
    return;
  dlarft_dtrmv_sm32x32_device_mio(n, tau_array[batchid], T_array[batchid], ldt);

}

__global__ void 
sof_dlarft_dtrmv_sm32x32_kernel_batched_FACT(int n,
					     tipo **tau_array, tipo **T_array, int ldt,
					     int nb_slices,
					     int slice_size,
					     int offset, tipo_int batchCount )
{
  
  int sliceId = blockIdx.z * blockDim.z; 
  int batchId_A = sliceId * slice_size + offset;

  // if ((threadIdx.x == 0) && (threadIdx.y == 0))
  //   printf("block = %d,  sliceId = %d, batchId_A = %d \n", blockIdx.z, sliceId, batchId_A );
  
  if ((sliceId >= nb_slices) || (batchId_A >= batchCount))
    return;
  dlarft_dtrmv_sm32x32_device_mio(n, tau_array[sliceId], T_array[batchId_A], ldt);

}

/******************************************************************************/
__global__ void dlarft_dtrmv_sm32x32_kernel_mio(int n, tipo *tau, tipo *T, int ldt )
{  
    dlarft_dtrmv_sm32x32_device_mio(n, tau, T, ldt);
}

/******************************************************************************/
static __device__ void 
dlarft_recdtrmv_sm32x32_device(
    int m, int n, tipo *tau,
    tipo *Trec, int ldtrec, tipo *Ttri, int ldttri)
{
    int tx = threadIdx.x; 
    tipo *sdata = (tipo*) shared_data;
    tipo res;

    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate 
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n; s++)
    {
        sdata[tx + s*n] = Trec[tx + s * ldtrec];
    }
    // __syncthreads();  
    
    // perform sequence of n-1 gemv
    for (int i=0; i < n; i++)
    {
        res = d_zero;
        for (int j=0; j < i; j++)
        {
            res += sdata[tx + j * n] * Ttri[j+ i * ldttri];      
        }
        // __syncthreads();   // a enlever
        sdata[tx + i * n] = -tau[i] * (sdata[tx + i * n] + res);
        // __syncthreads();  
    } 

    // write back the updated block of k column of T  multiplying by -tau
    for (int s=0; s < n; s++)
    {
        Trec[tx + s * ldtrec] = sdata[tx + s*n];
    }
}


/******************************************************************************/
__global__ void 
dlarft_recdtrmv_sm32x32_kernel_batched(
    int m, int n, tipo **tau_array,
    tipo **Trec_array, int ldtrec, tipo **Ttri_array, int ldttri)
{
    int batchId = blockIdx.z;
    dlarft_recdtrmv_sm32x32_device(m, n, tau_array[batchId], Trec_array[batchId], ldtrec, Ttri_array[batchId], ldttri);
}

/******************************************************************************/
extern "C"
void magmablas_dlarft_recdtrmv_sm32x32_batched(
    int m, int n, 
    tipo **tau_array, 
    tipo **Trec_array, int ldtrec, 
    tipo **Ttri_array, int ldttri,
    tipo_int batchCount)
{
    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(tipo)*(m*n);
    dlarft_recdtrmv_sm32x32_kernel_batched
      <<< grid, threads, shmem>>>
        (m, n,  tau_array, Trec_array, ldtrec, Ttri_array, ldttri);
}

extern "C" int
magma_larft_batched(int n, int k,
		     tipo **v_array, int ldv,
		     tipo **tau_array, tipo **T_array, int ldt, 
		     tipo **work_array, int lwork, 
		     tipo_int batchCount)
{
  /* 
n The order of the block reflector H.
k The order of the triangular factor T, is equal to the number of elementary reflectors.
*/

    if ( k <= 0) return 0;

    int maxnb = max_shared_bsiz;

    int DEBUG=0;
    int nb = min(k,maxnb);

    int i, j, prev_n, mycol, rows;

    tipo **dW1_displ  = NULL;
    tipo **dW2_displ  = NULL;
    tipo **dW3_displ  = NULL;
    tipo **dTstep_array  = NULL;

    cudaMalloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    cudaMalloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    cudaMalloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    cudaMalloc((void**)&dTstep_array,  batchCount * sizeof(*dTstep_array));

    if (k > nb) //suddivido la matrice in blocchi da nb colonne
    {
       //dTstep_array[i] = & work_array[i][0*lwork+0], i = 0,...,batchCount-1
        magma_ddisplace_pointers(dTstep_array, work_array, lwork, 0, 0, batchCount );
    }
    else // c'e' un unico blocco di colonne
    {
      //dTstep_array[i] = & T_array[i][0*ldt+0], i = 0,...,batchCount-1
        magma_ddisplace_pointers(dTstep_array, T_array, ldt, 0, 0, batchCount );
    }

    int ldtstep = ldt; //a enlever
   

    // GEMV compute the whole triangular upper portion of T (phase 1)
    // calcola T cappuccio
    // magma_gemm_batched( 'T', 'N', 
    //                      k, k, n, 
    //                      c_one,  v_array, ldv, 
    //                              v_array, ldv, 
    //                      c_zero, dTstep_array, ldtstep, 
    //                      batchCount  );
#ifdef DOUBLE_PRECISION
    cublasDgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		       int(k), int(k), int(n),
		       &one, (const tipo**)v_array, int(ldv),
		       (const tipo**)v_array, int(ldv),
		       &zero, dTstep_array, int(ldtstep), batchCount );
#else
    cublasSgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		       int(k), int(k), int(n),
		       &one, (const tipo**)v_array, int(ldv),
		       (const tipo**)v_array, int(ldv),
		       &zero, dTstep_array, int(ldtstep), batchCount );
#endif
    //inizializza la matrice a zero nella parte triangolare e la diagonale a zero
    //magmablas_dlaset_batched( MagmaLower, k, k, MAGMA_D_ZERO, MAGMA_D_ZERO, dTstep_array, ldtstep, batchCount  );
    
    // no need for it as T is expected to be lower zero
    //if (k > nb) magmablas_dlaset_batched( MagmaLower, k, k, MAGMA_D_ZERO, MAGMA_D_ZERO, dTstep_array, ldtstep, batchCount  );
    

    //TRMV
    //T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    // TRMV is split over block of column of size nb 
    // the update should be done from top to bottom so:
    // 1- a gemm using the previous computed columns
    //    of T to update rectangular upper protion above 
    //    the triangle of my columns 
    // 2- the columns need to be updated by a serial 
    //    loop over of gemv over itself. since we limit the
    //    shared memory to nb, this nb column 
    //    are split vertically by chunk of nb rows

    dim3 grid(1, 1, batchCount);

    // Finto for se k<nb
    for (j=0; j < k; j += nb)
    {
        prev_n =  j;
        mycol  =  min(nb, k-j); // = k se k<nb
        // note that myrow = prev_n + mycol;
	
	// inizio IF (se k < nb, npn si entra in questo if perche' prev_n = j = 0)
        if (prev_n > 0 && mycol > 0) {
            if (DEBUG == 3) {
                printf("doing gemm on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) prev_n, (long long) mycol, (long long) 0, (long long) j );
            }

	    //dW1_displ[i] = & Tstep_array[i][j*ldt+0], i = 0,...,batchCount-1
            magma_ddisplace_pointers(dW1_displ, dTstep_array, ldtstep, 0, j, batchCount);
	    //dW2_displ[i] = & T_array[i][j*ldt+0], i = 0,...,batchCount-1
            magma_ddisplace_pointers(dW2_displ, T_array,     ldt, 0, j, batchCount);
	    
            // magma_gemm_batched( MagmaNoTrans, MagmaNoTrans, 
            //                      prev_n, mycol, prev_n, 
            //                      c_one,  T_array, ldt, 
            //                              dW1_displ, ldtstep, 
            //                      c_zero, dW2_displ, ldt, 
            //                      batchCount );
#ifdef DOUBLE_PRECISION
	    cublasDgemmBatched(cublasHandle, cublas_trans, cublas_trans,
			       int(prev_n), int(mycol), int(prev_n),
			       &one, (const tipo**)T_array, int(ldt),
			       (const tipo**)dW1_displ, int(ldtstep),
			       &zero, dW2_displ, int(ldt), batchCount );
#else
	    cublasSgemmBatched(cublasHandle, cublas_trans, cublas_trans,
			       int(prev_n), int(mycol), int(prev_n),
			       &one, (const tipo**)T_array, int(ldt),
			       (const tipo**)dW1_displ, int(ldtstep),
			       &zero, dW2_displ, int(ldt), batchCount );
#endif

            // update my rectangular portion (prev_n,mycol) using sequence of gemv
	    //dW1_displ[i] = & Tstep_array[i][j*ldt+j], i = 0,...,batchCount-1
            magma_ddisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount);
	    //dW3_displ[i] = & tau_array[i][j*1+0], i = 0,...,batchCount-1
            magma_ddisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount);

            for (i=0; i < prev_n; i += nb)
            {
                rows = min(nb,prev_n-i);
                if (DEBUG == 3) {
                    printf("        doing recdtrmv on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                            (long long) rows, (long long) mycol, (long long) i, (long long) j );
                }

                if (rows > 0 && mycol > 0)
                {
		  //dW2_displ[k] = & T_array[k][j*ldt+i], k = 0,...,batchCount-1
                    magma_ddisplace_pointers(dW2_displ, T_array,  ldt, i, j, batchCount);
                    magmablas_dlarft_recdtrmv_sm32x32_batched(rows, mycol, dW3_displ, dW2_displ, ldt, dW1_displ, ldtstep, batchCount);
                }
            }
        }
	// fine IF

        // the upper rectangular portion is updated, now if needed update the triangular portion
	// Commento l'if perche' dovrebbe essere: star_T = 0 
        //if (stair_T == 0) {
	if (DEBUG == 3) {
	  printf("doing dtrmv on the triangular portion of size %lld %lld of T(%lld,%lld)\n",
		 (long long) mycol, (long long) mycol, (long long) j, (long long) j );
	}

	if (mycol > 0)
	  {
	    //dW1_displ[i] = & Tstep_array[i][j*ldt+j], i = 0,...,batchCount-1
	    magma_ddisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount);
	    //dW3_displ[i] = & tau_array[i][0*ldt+j], i = 0,...,batchCount-1
	    magma_ddisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount );
	    //dW2_displ[i] = & T_array[i][j*ldt+j], i = 0,...,batchCount-1
	    magma_ddisplace_pointers(dW2_displ, T_array,     ldt, j, j, batchCount );
	    magmablas_dlarft_dtrmv_sm32x32_batched(mycol, mycol, dW3_displ, dW1_displ, ldtstep, dW2_displ, ldt, batchCount);
	  }
	   //}
    }// end of j

    cudaFree(dW1_displ);
    cudaFree(dW2_displ);
    cudaFree(dW3_displ);
    cudaFree(dTstep_array);

    return 0;
}

extern "C" int
magma_larft_sm32x32_batched(int m, int n,
			    tipo **V_array, int ldv,
			    tipo **Tau_array, tipo **T_array, int ldt, 
			    tipo_int batchCount)
{
  /* MIA FUNZIONE


n The order of the block reflector H.
k The order of the triangular factor T, is equal to the number of elementary reflectors.
*/

  if ( n <= 0) return 0;

  dim3 blocks(1, 1, batchCount);
  dim3 threads(BLOCK_SIZE);
  
  // Y cappuccio
#ifdef DOUBLE_PRECISION
  cublasStatus1 = cublasDgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
				     int(n), int(n), int(m),
				     &one,
				     (const tipo**)V_array, int(ldv),
				     (const tipo**)V_array, int(ldv),
				     &zero,
				     T_array, int(ldt),
				     batchCount );
#else
  cublasStatus1 = cublasSgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
				     int(n), int(n), int(m),
				     &one,
				     (const tipo**)V_array, int(ldv),
				     (const tipo**)V_array, int(ldv),
				     &zero,
				     T_array, int(ldt), batchCount );
#endif
  
  if (cublasStatus1 != CUBLAS_STATUS_SUCCESS)
    printf("ERROR: cublas<t>gemmBatched failed \n");
  // else
  //   printf("cublas<t>gemmBatched succeded \n");

  // Aggiorna le colonne di T
  size_t shmem = sizeof(tipo)*(n*n);
  dlarft_dtrmv_sm32x32_kernel_batched_mio<<< blocks, threads, shmem >>> (n, Tau_array, T_array, ldt,  batchCount);

  
  return 0;
}


extern "C" void
parasof_larfb_gemm_batched(int m, int n, tipo **dA_array, int lda, tipo **dV_array, int ldv, tipo **dT_array, int ldt, tipo_int batchCount)
{

  tipo **dV2_displ  = NULL;
  tipo **dA_displ  = NULL;
  //tipo **dA_displ_dx  = NULL; forse si possono separare gli aggiornamenti dx e sx con gli streams!

  cudaMalloc((void**)&dV2_displ,  batchCount * sizeof(*dV2_displ));
  cudaMalloc((void**)&dA_displ,  batchCount * sizeof(*dA_displ));

  tipo **d_work1 = 0;
  tipo **d_work1_dev = NULL;
  tipo **d_work2 = 0;
  tipo **d_work2_dev = NULL;

  d_work1 = (tipo**)malloc(batchCount*sizeof(*d_work1));
  d_work2 = (tipo**)malloc(batchCount*sizeof(*d_work2));
 
  for (int i = 0; i < batchCount; i++){
    cudaMalloc((void**)&d_work1[i], (n*n)*sizeof(tipo));
    cudaMalloc((void**)&d_work2[i], (n*n)*sizeof(tipo));
  }

  cudaMalloc((void**)&d_work1_dev,  batchCount*sizeof(tipo*));
  cudaMalloc((void**)&d_work2_dev,  batchCount*sizeof(tipo*));
  cudaMemcpy(d_work1_dev, d_work1, batchCount * sizeof(tipo*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_work2_dev, d_work2, batchCount * sizeof(tipo*), cudaMemcpyHostToDevice);
  

  //Puntatori a V2 // row = n, column = 0, batchId = batchId
  zdisplace_pointers_kernel<<< batchCount, 1>>>(dV2_displ, dV_array, ldv, n, 0);
  
  //SPOSTA I PUNTATORI A SX // row = n, column = 0, batchId -= 1
  sx_displace_pointers_kernel<<< batchCount, 1>>>(dA_displ, dA_array, lda, n, 0, batchCount);

  // work1 = V1.T*A
#ifdef DOUBLE_PRECISION
  cublasDgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one,
		     (const tipo**)dV_array, int(ldv),
		     (const tipo**)dA_displ, int(lda),
		     &zero, d_work1_dev, int(n), batchCount );
#else
  cublasSgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one, (const tipo**)dV_array, int(ldv),
		     (const tipo**)dA_displ, int(lda),
		     &zero, d_work1_dev, int(n), batchCount );
#endif

  // work2 = T.T*work1
#ifdef DOUBLE_PRECISION
  cublasDgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one,
		     (const tipo**)dT_array, int(ldt),
		     (const tipo**)d_work1_dev, int(lda),
		     &zero, d_work2_dev, int(n), batchCount );
#else
  cublasSgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one,
		     (const tipo**)dT_array, int(ldt),
		     (const tipo**)d_work1_dev, int(lda),
		     &zero, d_work2_dev, int(n), batchCount );
#endif

  // dA_displ = V2*work2
#ifdef DOUBLE_PRECISION
  cublasDgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one,
		     (const tipo**)dV2_displ, int(ldv),
		     (const tipo**)d_work2_dev, int(n),
		     &zero, dA_displ, int(lda), batchCount );
#else
  cublasSgemmBatched(cublasHandle, cublas_trans_t, cublas_trans,
		     int(n), int(n), int(n),
		     &one,
		     (const tipo**)dV2_displ, int(ldv),
		     (const tipo**)d_work2_dev, int(n),
		     &zero, dA_displ, int(lda), batchCount );
#endif
  
  //SPOSTA I PUNTATORI A DX // row = 0, column = 0, batchId += 1
  dx_displace_pointers_kernel<<< batchCount, 1>>>(dA_displ, dA_array, lda, 0, 0, batchCount);

  for(int i=0; i<batchCount; i++) {
    cudaFree(d_work1[i]);
    cudaFree(d_work2[i]);
   }
  cudaFree(d_work1_dev);
  cudaFree(d_work2_dev);
  free(d_work1);
  free(d_work2);
  
  cudaFree(dV2_displ);
  cudaFree(dA_displ);
  
  return;
}
