#include "sparse_defs.hpp"
#include "cg.hpp"

extern "C" {

//static int   kernel_colptr[N+1];
//static int   kernel_col[NV];
//static float kernel_val[NV];
static float kernel_b[N];
static float kernel_x[N];

static float p[N];
static float q[N];
static float r[N];

void kernel_func
(//int colptr[N+1],
 //int col[NV],
 //float val[NV],
 float b[N],
 float x[N],
 int max_iter,
 int iter[1],
 float tol_error,
 float ret_error[1]
 )
{
#pragma HLS ARRAY_PARTITION variable=kernel_b cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=kernel_x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=kernel_b dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=kernel_x dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=r dim=1 complete
  int tmp_iter;
  float tmp_error;
  /*
  copy_colptr:
  for (int i=0; i<N+1; i++) {
    kernel_colptr[i] = colptr[i];
  }
  copy_col_val:
  for (int i=0; i<NV; i++) {
    kernel_col[i] = col[i];
    kernel_val[i] = val[i];
  }
  */
  copy_b:
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
    kernel_b[i] = b[i];
  }
  solve<float>
    (//kernel_colptr, kernel_col, kernel_val,
     kernel_b, kernel_x,
     max_iter, tmp_iter,
     tol_error, tmp_error,
     p, q, r);
  iter[0] = tmp_iter;
  ret_error[0] = tmp_error;
  copy_x:
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
    x[i] = kernel_x[i];
  }
  return;
}

} /* end of extern C */
