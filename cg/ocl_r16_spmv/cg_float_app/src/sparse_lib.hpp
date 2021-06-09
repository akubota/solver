#ifndef __INCLUDED_SPARSE_LIB_HPP__
#define __INCLUDED_SPARSE_LIB_HPP__

#include "sparse_defs.hpp"

template<typename T>
T
dot(T a[N], T b[N])
{
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=a dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=b dim=1 complete
  int i;
  T sum = (T)0;
  dot:
  for (i=0; i<N; i++) {
//#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
    sum += a[i]*b[i];
  }
  return sum;
}

template<typename T>
T
r16_dot_rd
(T p[N], T q[N])
{
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
  T sum;
  T s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
  T s01, s23, s45, s67, s89, s1011, s1213, s1415;
  T s0123, s4567, s891011, s12131415;
  T s01234567, s89101112131415;

  s0 = p[0] * q[0];
  s1 = p[1] * q[1];
  s01 = s0 + s1;

  s2 = p[2] * q[2];
  s3 = p[3] * q[3];
  s23 = s2 + s3;
  s0123 = s01 + s23;

  s4 = p[4] * q[4];
  s5 = p[5] * q[5];
  s45 = s4 + s5;

  s6 = p[6] * q[6];
  s7 = p[7] * q[7];
  s67 = s6 + s7;
  s4567 = s45 + s67;
  s01234567 = s0123 + s4567;

  s8 = p[8] * q[8];
  s9 = p[9] * q[9];
  s89 = s8 + s9;

  s10 = p[10] * q[10];
  s11 = p[11] * q[11];
  s1011 = s10 + s11;
  s891011 = s89 + s1011;

  s12 = p[12] * q[12];
  s13 = p[13] * q[13];
  s1213 = s12 + s13;

  s14 = p[14] * q[14];
  s15 = p[15] * q[15];
  s1415 = s14 + s15;
  s12131415 = s1213 + s1415;
  s89101112131415 = s891011 + s12131415;
  sum = s01234567 + s89101112131415;

  return sum;
}

template<typename T>
void
spmv(int colptr[N+1], int col[NV], T val[NV],
     T b[N], T ret[N])
{
  int i, j;
  int cb, ce;

  spmv:
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
r16_spmv
(T p[N], T q[N])
{
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete

  q[0] = p[0] -0.5*p[2];
  q[1] = 1.5*p[1] - p[2] - p[3];
  q[2] = -0.5*p[0] - p[1] + 1.5*p[2] - 0.5*p[4] - 0.5*p[5];
  q[3] = -p[1] + 1.5*p[3];
  q[4] = -0.5*p[2] + 1.5*p[4] - p[6];
  q[5] = -0.5*p[2] + 1.5*p[5] - 0.5*p[8];
  q[6] = -p[4] + 1.5*p[6] - 0.5*p[7] -0.5*p[9];
  q[7] = -0.5*p[6] + p[7];
  q[8] = -0.5*p[5] + p[8] - 0.5*p[10];
  q[9] = -0.5*p[6] + 1.5*p[9];
  q[10] = -0.5*p[8] + 1.5*p[10] - p[11] - 0.5*p[12] - 0.5*p[13];
  q[11] = -p[10] + 1.5*p[11];
  q[12] = -0.5*p[10] + 1.5*p[12] - p[14];
  q[13] = -0.5*p[10] + 1.5*p[13];
  q[14] = -p[12] + 1.5*p[14] -0.5*p[15];
  q[15] = -0.5*p[14] + 1.5*p[15];
  return;
}

template<typename T>
T
spmv_dot
(int colptr[N+1], int col[NV], T val[NV],
 T b[N], T ret[N])
{
  int i, j;
  int cb, ce;
  T sum = (T)0;

  spmv_dot1:
  for (i=0; i<N; i++) {
    cb = colptr[i];
    ce = colptr[i+1];
    ret[i] = (T)0;
    spmv_dot2:
    for (j=cb; j<ce; j++) {
      ret[i] += val[j] * b[col[j]];
    }
    sum += b[i] * ret[i];
  }
  return sum;
}

template<typename T>
T
r16_spmv_dot
(T p[N], T q[N])
{
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete

  T sum = (T)0;

  q[0] = p[0] -0.5*p[2];
  sum += (p[0] * q[0]);

  q[1] = 1.5*p[1] - p[2] - p[3];
  sum += (p[1] * q[1]);

  q[2] = -0.5*p[0] - p[1] + 1.5*p[2] - 0.5*p[4] - 0.5*p[5];
  sum += (p[2] * q[2]);

  q[3] = -p[1] + 1.5*p[3];
  sum += (p[3] * q[3]);

  q[4] = -0.5*p[2] + 1.5*p[4] - p[6];
  sum += (p[4] * q[4]);

  q[5] = -0.5*p[2] + 1.5*p[5] - 0.5*p[8];
  sum += (p[5] * q[5]);

  q[6] = -p[4] + 1.5*p[6] - 0.5*p[7] -0.5*p[9];
  sum += (p[6] * q[6]);

  q[7] = -0.5*p[6] + p[7];
  sum += (p[7] * q[7]);

  q[8] = -0.5*p[5] + p[8] - 0.5*p[10];
  sum += (p[8] * q[8]);

  q[9] = -0.5*p[6] + 1.5*p[9];
  sum += (p[9] * q[9]);

  q[10] = -0.5*p[8] + 1.5*p[10] - p[11] - 0.5*p[12] - 0.5*p[13];
  sum += (p[10] * q[10]);

  q[11] = -p[10] + 1.5*p[11];
  sum += (p[11] * q[11]);

  q[12] = -0.5*p[10] + 1.5*p[12] - p[14];
  sum += (p[12] * q[12]);

  q[13] = -0.5*p[10] + 1.5*p[13];
  sum += (p[13] * q[13]);

  q[14] = -p[12] + 1.5*p[14] -0.5*p[15];
  sum += (p[14] * q[14]);

  q[15] = -0.5*p[14] + 1.5*p[15];
  sum += (p[15] * q[15]);

  return sum;
}

template<typename T>
T
r16_spmv_dot_tmp
(T p[N], T q[N])
{
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete

  T sum = (T)0;
  T p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  T q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15;
  p0 = p[0];
  p1 = p[1];
  p2 = p[2];
  p3 = p[3];
  p4 = p[4];
  p5 = p[5];
  p6 = p[6];
  p7 = p[7];
  p8 = p[8];
  p9 = p[9];
  p10 = p[10];
  p11 = p[11];
  p12 = p[12];
  p13 = p[13];
  p14 = p[14];
  p15 = p[15];

  q0 = p0 -0.5*p2;
  sum += (p0 * q0);
  q[0] = q0;

  q1 = 1.5*p1 - p2 - p3;
  sum += (p1 * q1);
  q[1] = q1;

  q2 = -0.5*p0 - p1 + 1.5*p2 - 0.5*p4 - 0.5*p5;
  sum += (p2 * q2);
  q[2] = q2;

  q3 = -p1 + 1.5*p3;
  sum += (p3 * q3);
  q[3] = q3;

  q4 = -0.5*p2 + 1.5*p4 - p6;
  sum += (p4 * q4);
  q[4] = q4;

  q5 = -0.5*p2 + 1.5*p5 - 0.5*p8;
  sum += (p5 * q5);
  q[5] = q5;

  q6 = -p4 + 1.5*p6 - 0.5*p7 -0.5*p9;
  sum += (p6 * q6);
  q[6] = q6;

  q7 = -0.5*p6 + p7;
  sum += (p7 * q7);
  q[7] = q7;

  q8 = -0.5*p5 + p8 - 0.5*p10;
  sum += (p8 * q8);
  q[8] = q8;

  q9 = -0.5*p6 + 1.5*p9;
  sum += (p9 * q9);
  q[9] = q[9];

  q10 = -0.5*p8 + 1.5*p10 - p11 - 0.5*p12 - 0.5*p13;
  sum += (p10 * q10);
  q[10] = q10;

  q11 = -p10 + 1.5*p11;
  sum += (p11 * q11);
  q[11] = q11;

  q12 = -0.5*p10 + 1.5*p12 - p14;
  sum += (p12 * q12);
  q[12] = q12;

  q13 = -0.5*p10 + 1.5*p13;
  sum += (p13 * q13);
  q[13] = q13;

  q14 = -p12 + 1.5*p14 -0.5*p15;
  sum += (p14 * q14);
  q[14] = q14;

  q15 = -0.5*p14 + 1.5*p15;
  sum += (p15 * q15);
  q[15] = q15;

  return sum;
}

template<typename T>
T
r16_spmv_dot_rd
(T p[N], T q[N])
{
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete

  T sum;
  T p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  T q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15;
  T s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
  T s01, s23, s45, s67, s89, s1011, s1213, s1415;
  T s0123, s4567, s891011, s12131415;
  T s01234567, s89101112131415;

  p0 = p[0];
  p1 = p[1];
  p2 = p[2];
  p3 = p[3];
  p4 = p[4];
  p5 = p[5];
  p6 = p[6];
  p7 = p[7];
  p8 = p[8];
  p9 = p[9];
  p10 = p[10];
  p11 = p[11];
  p12 = p[12];
  p13 = p[13];
  p14 = p[14];
  p15 = p[15];

  q0 = p0 -0.5*p2;
  s0 = p0 * q0;
  q[0] = q0;

  q1 = 1.5*p1 - p2 - p3;
  s1 = p1 * q1;
  s01 = s0 + s1;
  q[1] = q1;

  q2 = -0.5*p0 - p1 + 1.5*p2 - 0.5*p4 - 0.5*p5;
  s2 = p2 * q2;
  q[2] = q2;

  q3 = -p1 + 1.5*p3;
  s3 = p3 * q3;
  s23 = s2 + s3;
  s0123 = s01 + s23;
  q[3] = q3;

  q4 = -0.5*p2 + 1.5*p4 - p6;
  s4 = p4 * q4;
  q[4] = q4;

  q5 = -0.5*p2 + 1.5*p5 - 0.5*p8;
  s5 = p5 * q5;
  s45 = s4 + s5;
  q[5] = q5;

  q6 = -p4 + 1.5*p6 - 0.5*p7 -0.5*p9;
  s6 = p6 * q6;
  q[6] = q6;

  q7 = -0.5*p6 + p7;
  s7 = p7 * q7;
  s67 = s6 + s7;
  s4567 = s45 + s67;
  s01234567 = s0123 + s4567;
  q[7] = q7;

  q8 = -0.5*p5 + p8 - 0.5*p10;
  s8 = p8 * q8;
  q[8] = q8;

  q9 = -0.5*p6 + 1.5*p9;
  s9 = p9 * q9;
  s89 = s8 + s9;
  q[9] = q[9];

  q10 = -0.5*p8 + 1.5*p10 - p11 - 0.5*p12 - 0.5*p13;
  s10 = p10 * q10;
  q[10] = q10;

  q11 = -p10 + 1.5*p11;
  s11 = p11 * q11;
  s1011 = s10 + s11;
  s891011 = s89 + s1011;
  q[11] = q11;

  q12 = -0.5*p10 + 1.5*p12 - p14;
  s12 = p12 * q12;
  q[12] = q12;

  q13 = -0.5*p10 + 1.5*p13;
  s13 = p13 * q13;
  s1213 = s12 + s13;
  q[13] = q13;

  q14 = -p12 + 1.5*p14 -0.5*p15;
  s14 = p14 * q14;
  q[14] = q14;

  q15 = -0.5*p14 + 1.5*p15;
  s15 = p15 * q15;
  s1415 = s14 + s15;
  s12131415 = s1213 + s1415;
  s89101112131415 = s891011 + s12131415;
  sum = s01234567 + s89101112131415;
  q[15] = q15;

  return sum;
}

template<typename T>
void
axpy(T a, T x[N], T y[N], T ret[N])
{
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=ret cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=x dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=y dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=ret dim=1 complete

  axpy:
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
    ret[i] = a * x[i] + y[i];
  }
  return;
}

template<typename T>
void
axpy2(T alpha, T p[N], T x[N], T q[N], T r[N])
{
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=x dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=p dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=r dim=1 complete

  axpy2:
  for (int i=0; i<N; i++) {
#pragma HLS UNROLL factor=8
//#pragma HLS UNROLL
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
  axpy2_dot:
  for (int i=0; i<N; i++) {
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
    sum += r[i] * r[i];
  }
  return sum;
}

template<typename T>
T
r16_axpy2_dot_rd
(T alpha, T p[N], T x[N], T q[N], T r[N])
{
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=x dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=q dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=r dim=1 complete

  T sum;
  T s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
  T s01, s23, s45, s67, s89, s1011, s1213, s1415;
  T s0123, s4567, s891011, s12131415;
  T s01234567, s89101112131415;

  x[0] = x[0] + alpha * p[0];
  r[0] = r[0] - alpha * q[0];
  s0 = r[0] * r[0];
  x[1] = x[1] + alpha * p[1];
  r[1] = r[1] - alpha * q[1];
  s1 = r[1] * r[1];
  s01 = s0 + s1;

  x[2] = x[2] + alpha * p[2];
  r[2] = r[2] - alpha * q[2];
  s2 = r[2] * r[2];
  x[3] = x[3] + alpha * p[3];
  r[3] = r[3] - alpha * q[3];
  s3 = r[3] * r[3];
  s23 = s2 + s3;
  s0123 = s01 + s23;

  x[4] = x[4] + alpha * p[4];
  r[4] = r[4] - alpha * q[4];
  s4 = r[4] * r[4];
  x[5] = x[5] + alpha * p[5];
  r[5] = r[5] - alpha * q[5];
  s5 = r[5] * r[5];
  s45 = s4 + s5;

  x[6] = x[6] + alpha * p[6];
  r[6] = r[6] - alpha * q[6];
  s6 = r[6] * r[6];
  x[7] = x[7] + alpha * p[7];
  r[7] = r[7] - alpha * q[7];
  s7 = r[7] * r[7];
  s67 = s6 + s7;
  s4567 = s45 + s67;
  s01234567 = s0123 + s4567;

  x[8] = x[8] + alpha * p[8];
  r[8] = r[8] - alpha * q[8];
  s8 = r[8] * r[8];
  x[9] = x[9] + alpha * p[9];
  r[9] = r[9] - alpha * q[9];
  s9 = r[9] * r[9];
  s89 = s8 + s9;

  x[10] = x[10] + alpha * p[10];
  r[10] = r[10] - alpha * q[10];
  s10 = r[10] * r[10];
  x[11] = x[11] + alpha * p[11];
  r[11] = r[11] - alpha * q[11];
  s11 = r[11] * r[11];
  s1011 = s10 + s11;
  s891011 = s89 + s1011;

  x[12] = x[12] + alpha * p[12];
  r[12] = r[12] - alpha * q[12];
  s12 = r[12] * r[12];
  x[13] = x[13] + alpha * p[13];
  r[13] = r[13] - alpha * q[13];
  s13 = r[13] * r[13];
  s1213 = s12 + s13;

  x[14] = x[14] + alpha * p[14];
  r[14] = r[14] - alpha * q[14];
  s14 = r[14] * r[14];
  x[15] = x[15] + alpha * p[15];
  r[15] = r[15] - alpha * q[15];
  s15 = r[15] * r[15];
  s1415 = s14 + s15;
  s12131415 = s1213 + s1415;
  s89101112131415 = s891011 + s12131415;
  sum = s01234567 + s89101112131415;

  return sum;
}

#endif
