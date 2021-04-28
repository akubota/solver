#include "sparse_defs.hpp"
#include "sparse_lib.hpp"
#include "util_host.hpp"
#include "cg.hpp"
#include <fstream>
#include <iostream>

#define USE_TIMESPEC

#ifdef USE_TIMESPEC
#include <time.h>
#else
#include <sys/time.h>
#endif

int rowptr[N+1];
int row[NV];
float ccs_val[NV];
int colptr[N+1];
int col[NV];
float crs_val[NV];
float diag[N];
float b[N];
float x[N];

float p[N];
float q[N];
float r[N];

int main(int argc, char *argv[])
{
  float res;
  float rhsNorm2;
  int iter;
  float ret_error;
#ifdef USE_TIMESPEC
  struct timespec t1, t2;
#else
  struct timeval t1, t2;
#endif

  // default parameters
  const int default_max_iter = NITER;
  const float default_tol_error = 1.0e-4;
  const char *default_coef_filename = "sym_sparse.mtx";

  // set parameters if specified
  int max_iter = default_max_iter;
  float tol_error = default_tol_error;
  const char * coef_filename = default_coef_filename;
  parseArgsGetOpt<float>(argc, argv, max_iter, tol_error, &coef_filename);
  
  int tmp_rows, tmp_cols, tmp_nnzs;
  bool issymmetric;

  std::ifstream coef_file;
  coef_file.open(coef_filename, std::ifstream::in);
  if (!coef_file) {
    std::cerr << "File open error: " << coef_filename << std::endl;
    return 1;
  }

  readHeaders(coef_file, tmp_rows, tmp_cols, tmp_nnzs, issymmetric);
  if (!issymmetric) {
    std::cerr << "Symmetric matrix is required\n";
    return 2;
  }
  mmtoccs<float>
    (coef_file, rowptr, row, ccs_val, diag, tmp_nnzs, tmp_cols);
  coef_file.close();

  symccstocrs<float>
    (colptr, col, crs_val,
     rowptr, row, ccs_val,
     tmp_rows, tmp_cols);

  init_rhs<float>(colptr, col, crs_val, x, b);

  rhsNorm2 = dot<float>(b, b);
  if (rhsNorm2 == (float)0) {
    std::cerr << "RHS vector is zero\n";
    return 3;
  }
  
#ifdef USE_TIMESPEC
  clock_gettime(CLOCK_REALTIME, &t1);
#else
  gettimeofday(&t1, NULL);
#endif

  solve<float>
    (colptr, col, crs_val,
     b, x,
     max_iter, iter,
     tol_error, ret_error,
     p, q, r);

#ifdef USE_TIMESPEC
  clock_gettime(CLOCK_REALTIME, &t2);
#else
  gettimeofday(&t2, NULL);
#endif

  res = calc_res<float>(colptr, col, crs_val, x, b);
  print_stderr_res<float>(iter, res);
  print_stderr_error<float>(ret_error);
  print_vec<float>(x);

#ifdef USE_TIMESPEC
  std::cerr << "Time: " <<
    t2.tv_sec + t2.tv_nsec*1.0e-9 -
    t1.tv_sec - t1.tv_nsec*1.0e-9 << std::endl;
#else
  std::cerr << "Time: " <<
    t2.tv_sec + t2.tv_usec*1.0e-6 -
    t1.tv_sec - t1.tv_usec*1.0e-6 << std::endl;
#endif

  return 0;
}
