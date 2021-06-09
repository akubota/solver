#ifndef __INCLUDED_CG_HPP__
#define __INCLUDED_CG_HPP__

#include "sparse_defs.hpp"
#include "sparse_lib.hpp"
#include <cmath>
#include <limits>     // std::numeric_limits
#include <algorithm>  // std::max
#include <iostream>

#define FIXEDSIZE

template <typename T>
void
solve
(//int colptr[N+1],
 //int col[NV],
 //T val[NV],
 T b[N],
 T x[N],
 int max_iter,
 int & iter,
 T tol_error,
 T & ret_error,
 T p[N],
 T q[N],
 T r[N]
 )
{
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=b dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=x dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=r dim=1 complete
  T rho_old, rho_new, alpha, beta, rhsNorm2, resNorm2;
  T tol = tol_error;

  rhsNorm2 = r16_dot_rd<T>(b, b);
  //rhsNorm2 = dot<T>(b, b);
  init_x:
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
    x[i] = (T)0;
    r[i] = b[i];
    p[i] = b[i];
  }
  rho_new = rhsNorm2; // assume x=0

  //r16_spmv<T>(p, q);
  //alpha = rho_new/r16_dot_rd<T>(p, q);
  alpha = rho_new/r16_spmv_dot_rd<T>(p, q);
  //alpha = rho_new/dot<T>(p, q);
  //alpha = rho_new/spmv_dot<T>(colptr, col, val, p, q);
  axpy2<T>(alpha, p, x, q, r);
  resNorm2 = r16_dot_rd<T>(r, r);
  //resNorm2 = r16_axpy2_dot_rd<T>(alpha, p, x, q, r);
  //resNorm2 = dot<T>(r, r);
  //resNorm2 = axpy2_dot<T>(alpha, p, x, q, r);

#ifndef FIXEDSIZE
  // cf. Eigen/src/IterativeSolver/ConjugateGradient.h
  const T considerAsZero = (std::numeric_limits<T>::min)();
  const T threshold = std::max<T>(T(tol*tol*rhsNorm2), considerAsZero);
#endif

#ifdef FIXEDSIZE
  for (iter=1; iter<NITER; iter++) {
#else
  for (iter=1; (iter<max_iter)&&(resNorm2>threshold); iter++) {
#endif

    rho_old = rho_new;
    rho_new = resNorm2;
    beta = rho_new/rho_old;
    axpy<T>(beta, p, r, p);
    //r16_spmv<T>(p, q);
    //alpha = rho_new/r16_dot_rd<T>(p, q);
    alpha = rho_new/r16_spmv_dot_rd<T>(p, q);
    //alpha = rho_new/dot<T>(p, q);
    //alpha = rho_new/spmv_dot<T>(colptr, col, val, p, q);
    axpy2<T>(alpha, p, x, q, r);
    resNorm2 = r16_dot_rd<T>(r, r);
    //resNorm2 = r16_axpy2_dot_rd<T>(alpha, p, x, q, r);
    //resNorm2 = dot<T>(r, r);
    //resNorm2 = axpy2_dot<T>(alpha, p, x, q, r);
  }
  ret_error = rho_new; // calculate sqrt(rho_new / rhsNorm2) in call site
  return;
}
  
#endif
