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
spmv_ell(int ell_col[N][NZ], T ell_val[N][NZ],
     T b[N], T ret[N])
{
  for (int i=0; i<N; i++) {
    ret[i] = (T)0;
    for (int j=0; j<NZ; j++) {
      ret[i] += ell_val[i][j] * b[ell_col[i][j]];
    }
  }
  return;
}

template<typename T>
T
spmv_ell_dot
( int ell_col[N][NZ], T ell_val[N][NZ],
  T b[N], T ret[N])
{
  T sum = (T)0;
  for (int i=0; i<N; i++) {
    ret[i] = (T)0;
    for (int j=0; j<NZ; j++) {
      ret[i] += ell_val[i][j] * b[ell_col[i][j]];
    }
    sum += b[i] * ret[i];
  }
  return sum;
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

template<typename T>
void
axpy2(T alpha, T p[N], T x[N], T q[N], T r[N])
{
  for (int i=0; i<N; i++) {
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
  }
  return;
}

template<typename T>
T
axpy2_dot(T alpha, T p[N], T x[N], T q[N], T r[N])
{
  T sum = (T)0;
  for (int i=0; i<N; i++) {
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
    sum += r[i] * r[i];
  }
  return sum;
}
#endif
