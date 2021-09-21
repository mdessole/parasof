/* get_config.cu -- modified by MAGMA http://icl.cs.utk.edu/magma/
   M. Dessole 21-09-2021
   Used in 
   "M. Dessole, F. Marcuzzi
   A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs
   Numerical Algorithms, 2020"
*/

#define NTCOL_1D_DEFAULT 32, 16, 10, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
const int ntcol_1d_default[] = {NTCOL_1D_DEFAULT};


#ifdef DOUBLE_PRECISION
// =============================================================================
// GEMM
// =============================================================================


// Kepler (or older) 
const int gemm_batched_ntcol_300[] = {64,32,32,8,5,8,10,8,6,5,4,2,3,1,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// Pascal (used also for maxwell) 
const int gemm_batched_ntcol_600[] = {64,64,32,16,16,8,10,1,8,8,2,2,3,5,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// Volta  
const int gemm_batched_ntcol_700[] = {64,15,32,15,15,8,10,8,8,10,8,2,3,5,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

// =============================================================================
// QR
// =============================================================================

// Kepler (or older)
const int geqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
// Pascal (used also for maxwell) 
const int geqrf_batched_ntcol_600[] = {16, 4, 3, 3, 5, 5, 5, 10, 14, 10, 10, 10, 12, 12, 10,  4, 6,  6, 4, 8, 7, 7, 4, 4, 6, 2, 2, 1, 2, 2, 1, 1};
// Volta 
const int geqrf_batched_ntcol_700[] = {32, 4, 2, 2, 1, 1, 1, 5, 9, 16, 11, 8, 3,  3, 15, 16,  8,  8,  8,  4, 12,  4,  4, 6, 6, 4, 6, 4, 6, 4, 4, 4};

#else
// =============================================================================
// GEMM
// =============================================================================

// Kepler (or older) 
const int gemm_batched_ntcol_300[] = {64,32,32,32,10,8,10,4,3,5,2,2,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// Pascal (used also for maxwell) 
const int gemm_batched_ntcol_600[] = {64,64,64,32,14,13,9,5,7,3,5,3,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// Volta  
const int gemm_batched_ntcol_700[] = {64,64,32,8,10,8,10,8,8,10,4,2,3,5,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1};


// =============================================================================
// QR
// =============================================================================

// Kepler (or older)
const int geqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT}
// Pascal (used also for maxwell) 
const int geqrf_batched_ntcol_600[] = {16, 4, 3, 4, 3, 3, 6, 14, 32, 32, 32, 32, 16, 14, 16, 10, 9, 10, 9, 9, 6, 8, 6, 6, 8, 8, 2, 6, 6, 2, 4, 2};
// Volta 
const int geqrf_batched_ntcol_700[] = {32, 4, 3, 6, 5, 3, 2, 2, 3,  3,  3, 3, 3, 11,  3, 16, 12, 12, 12, 12, 10, 10, 10, 8, 8, 8, 1, 1, 1, 1, 1, 1};
#endif



int magma_get_gemm_batched_ntcol(int m)
{
    int* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    ntcol_array = (int*)gemm_batched_ntcol_600;
    
    return ntcol_array[m-1];
}


int magma_get_geqr2_batched_ntcol(int m)
{
    int* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    ntcol_array = (int*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

// int magma_get_gemm_batched_ntcol(int m)
// {
//     int* ntcol_array; 

//     if(m < 0 || m > 32) return 1;
    
//     int arch = magma_getdevice_arch();
//     if      (arch <= 300) ntcol_array = (int*)gemm_batched_ntcol_300; 
//     else if (arch <= 600) ntcol_array = (int*)gemm_batched_ntcol_600;
//     else if (arch <= 700) ntcol_array = (int*)gemm_batched_ntcol_700;
//     else                  ntcol_array = (int*)ntcol_1d_default; 
    
//     return ntcol_array[m-1];
// }


// int magma_get_dgetrf_batched_ntcol(int m, int n)
// {
//     int* ntcol_array; 

//     if(m != n || m < 0 || m > 32) return 1;
    
//     int arch = magma_getdevice_arch();
//     if      (arch <= 300) ntcol_array = (int*)getrf_batched_ntcol_300; 
//     else if (arch <= 600) ntcol_array = (int*)getrf_batched_ntcol_600;
//     else if (arch <= 700) ntcol_array = (int*)getrf_batched_ntcol_700;
//     else                  ntcol_array = (int*)ntcol_1d_default; 
    
//     return ntcol_array[m-1];
// }

