#ifndef __INCLUDED_CG_HPP__
#define __INCLUDED_CG_HPP__

#include "sparse_defs.hpp"
#include "sparse_lib.hpp"
#include <cmath>
#include <limits>     // std::numeric_limits
#include <algorithm>  // std::max
#include <iostream>

template <typename T>
void
solve
(int colptr[N+1],
 int col[NV],
 T val[NV],
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
  T rho_old, rho_new, alpha, beta, rhsNorm2, resNorm2;
  T tol = tol_error;

  rhsNorm2 = dot<T>(b, b);
  resNorm2 = rhsNorm2; // assume x=0

  // cf. Eigen/src/IterativeSolver/ConjugateGradient.h
  const T considerAsZero = (std::numeric_limits<T>::min)();
  const T threshold = std::max<T>(T(tol*tol*rhsNorm2), considerAsZero);

  for (iter=0; (iter<max_iter)&&(resNorm2>threshold); iter++) {

    if (iter==0) {
      for (int i=0; i<N; i++) {
	x[i] = (T)0;
	r[i] = b[i];
	p[i] = b[i];
      }
      rho_new = resNorm2;
    } else {
      rho_old = rho_new;
      rho_new = resNorm2;
      beta = rho_new/rho_old;
      axpy<T>(beta, p, r, p);
    }
    spmv<T>(colptr, col, val, p, q);
    alpha = rho_new/dot<T>(p, q);
    axpy<T>(alpha, p, x, x);
    axpy<T>(-alpha, q, r, r);
    resNorm2 = dot<T>(r, r);
  }
  ret_error = std::sqrt(rho_new / rhsNorm2);
  return;
}
  
#endif
