#include "sparse_defs.hpp"
#include "cg.hpp"

extern "C" {

static int   kernel_ell_col[N][NZ];
static float kernel_ell_val[N][NZ];
static float kernel_b[N];
static float kernel_x[N];

static float p[N];
static float q[N];
static float r[N];

void kernel_func
(int ell_col[N*NZ],
 float ell_val[N*NZ],
 float b[N],
 float x[N],
 int max_iter,
 int iter[1],
 float tol_error,
 float ret_error[1]
 )
{
#pragma HLS ARRAY_PARTITION variable=kernel_ell_col cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=kernel_ell_val cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=kernel_b cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=kernel_x cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=r cyclic factor=16 dim=1
  int tmp_iter;
  float tmp_error;
  for (int i=0; i<N; i++) {
//#pragma HLS UNROLL factor=16
    for (int j=0; j<NZ; j++) {
      kernel_ell_col[i][j] = ell_col[i*NZ+j];
      kernel_ell_val[i][j] = ell_val[i*NZ+j];
    }    
  }
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=16
    kernel_b[i] = b[i];
  }
  solve<float>
    (kernel_ell_col, kernel_ell_val,
     kernel_b, kernel_x,
     max_iter, tmp_iter,
     tol_error, tmp_error,
     p, q, r);
  iter[0] = tmp_iter;
  ret_error[0] = tmp_error;
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=16
    x[i] = kernel_x[i];
  }
  return;
}

} /* end of extern C */
