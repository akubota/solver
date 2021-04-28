#include "sparse_defs.hpp"
#include "cg.hpp"

extern "C" {

static float p[N];
static float q[N];
static float r[N];

void kernel_func
(int colptr[N+1],
 int col[NV],
 float val[NV],
 float b[N],
 float x[N],
 int max_iter,
 int iter[1],
 float tol_error,
 float ret_error[1]
 )
{
  int tmp_iter;
  float tmp_error;
  solve<float>
    (colptr, col, val, b, x,
     max_iter, tmp_iter,
     tol_error, tmp_error,
     p, q, r);
  iter[0] = tmp_iter;
  ret_error[0] = tmp_error;
  return;
}

} /* end of extern C */
