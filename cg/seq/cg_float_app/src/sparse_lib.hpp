#ifndef __INCLUDED_SPARSE_LIB_HPP__
#define __INCLUDED_SPARSE_LIB_HPP__

#include "sparse_defs.hpp"

template<typename T>
T
dot(T a[N], T b[N])
{
  int i;
  T sum = (T)0;
  for (i=0; i<N; i++) {
    sum += a[i]*b[i];
  }
  return sum;
}

template<typename T>
void
spmv(int colptr[N+1], int col[NV], T val[NV],
     T b[N], T ret[N])
{
  int i, j;
  int cb, ce;

  for (i=0; i<N; i++) {
    cb = colptr[i];
    ce = colptr[i+1];
    ret[i] = (T)0;
    for (j=cb; j<ce; j++) {
      ret[i] += val[j] * b[col[j]];
    }
  }
  return;
}

template<typename T>
void
axpy(T a, T x[N], T y[N], T ret[N])
{
  for (int i=0; i<N; i++) {
    ret[i] = a * x[i] + y[i];
  }
  return;
}
#endif
