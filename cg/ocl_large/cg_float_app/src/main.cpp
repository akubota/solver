#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#include "sparse_defs.hpp"
#include "util_host.hpp"
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
int ell_col[N][NZ];
float ell_val[N][NZ];
float b[N];
float x[N];

extern "C" {

void kernel_func
(int ell_col[N][NZ],
 float ell_val[N][NZ],
 float b[N],
 float x[N],
 int max_iter,
 int iter[1],
 float tol_error,
 float ret_error[1]
 );

} /* end of extern "C" */

int main(int argc, char *argv[])
{
  float res;
  float rhsNorm2;
  int iter[1];
  float ret_error[1];
#ifdef USE_TIMESPEC
  struct timespec t1, t2;
#else
  struct timeval t1, t2;
#endif

  // default parameters
  const int default_max_iter = NITER;
  const float default_tol_error = 1.0e-4;
  const char * default_coef_filename = "sym_sparse.mtx";
  const char * default_xclbin_filename = "binary_container_1.xclbin";

  // set parameters if specified
  int max_iter = default_max_iter;
  float tol_error = default_tol_error;
  const char * coef_filename = default_coef_filename;
  const char * xclbin_filename = default_xclbin_filename;
  parseArgsGetOpt<float>
    (argc, argv, max_iter, tol_error, &coef_filename, &xclbin_filename);

  int tmp_rows, tmp_cols, tmp_nnzs;
  bool issymmetric;

  // Read and transform coefficient matrix data
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

  crstoellcrs<float>
    ((int *)ell_col, (float *)ell_val,
     colptr, col, crs_val,
     N, NZ);

  init_rhs<float>(colptr, col, crs_val, x, b);

  rhsNorm2 = dot_host<float>(b, b);
  if (rhsNorm2 == (float)0) {
    std::cerr << "RHS vector is zero\n";
    return 3;
  }

#ifdef USE_TIMESPEC
  clock_gettime(CLOCK_REALTIME, &t1);
#else
  gettimeofday(&t1, NULL);
#endif

  //////////////////////////////////////////////////////////////////////
  // OpenCL Setup
  //////////////////////////////////////////////////////////////////////

  // Device and platform
  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::Platform> platforms;
  bool found_device = false;
  cl::Platform::get(&platforms);
  for (size_t i=0; (i < platforms.size()) & (found_device == false); i++) {
    cl::Platform platform = platforms[i];
    std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
    if (platformName == "Xilinx") {
      devices.clear();
      platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
      if (devices.size()) {
	device = devices[0];
	found_device = true;
	break;
      }
    }
  }
  if (found_device == false) {
    std::cerr << "Error: Unable to find Target Device "
	      << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    return EXIT_FAILURE;
  }

  // Context and CommandQueue
  cl::Context context(device);
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Load xclbin
  std::cerr << "Loading: '" << xclbin_filename << "'\n";
  std::ifstream bin_file(xclbin_filename, std::ifstream::binary);
  if (!bin_file) {
    std::cerr << "OpenCL file open error: " << xclbin_filename << std::endl;
    return 1;
  }
  bin_file.seekg(0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg(0, bin_file.beg);
  char *buf = new char[nb];
  bin_file.read(buf, nb);

  // OpenCL Program
  cl::Program::Binaries bins;
  bins.push_back({buf,nb});
  devices.resize(1);
  cl::Program program(context, devices, bins);

  // Kernel on FPGA
  cl::Kernel krnl(program, "kernel_func");

  // cl::Buffer for kernel arguments
  size_t ell_nz_int_size_in_bytes = sizeof(int)   * (N*NZ);
  size_t ell_nz_val_size_in_bytes = sizeof(float) * (N*NZ);
  size_t vector_size_in_bytes     = sizeof(float) * (N);
  size_t int_size_in_bytes        = sizeof(int)   * (1);
  size_t float_size_in_bytes      = sizeof(float) * (1);

  cl::Buffer buffer_ell_col(context, CL_MEM_READ_ONLY, ell_nz_int_size_in_bytes);
  cl::Buffer buffer_ell_val(context, CL_MEM_READ_ONLY, ell_nz_val_size_in_bytes);
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, vector_size_in_bytes);
  cl::Buffer buffer_x(context, CL_MEM_WRITE_ONLY, vector_size_in_bytes);
  cl::Buffer buffer_iter(context, CL_MEM_WRITE_ONLY, int_size_in_bytes);
  cl::Buffer buffer_error(context, CL_MEM_WRITE_ONLY, float_size_in_bytes);

  int narg=0;
  krnl.setArg(narg++, buffer_ell_col);
  krnl.setArg(narg++, buffer_ell_val);
  krnl.setArg(narg++, buffer_b);
  krnl.setArg(narg++, buffer_x);
  krnl.setArg(narg++, max_iter);
  krnl.setArg(narg++, buffer_iter);
  krnl.setArg(narg++, tol_error);
  krnl.setArg(narg++, buffer_error);

  // pointers for accesing the buffers
  int * ptr_ell_col   = (int *) q.enqueueMapBuffer
    (buffer_ell_col, CL_TRUE, CL_MAP_WRITE, 0, ell_nz_int_size_in_bytes);
  float * ptr_ell_val = (float *) q.enqueueMapBuffer
    (buffer_ell_val, CL_TRUE, CL_MAP_WRITE, 0, ell_nz_val_size_in_bytes);
  float * ptr_b       = (float *) q.enqueueMapBuffer
    (buffer_b,       CL_TRUE, CL_MAP_WRITE, 0, vector_size_in_bytes);
  float * ptr_x       = (float *) q.enqueueMapBuffer
    (buffer_x,       CL_TRUE, CL_MAP_READ,  0, vector_size_in_bytes);
  int * ptr_iter      = (int *) q.enqueueMapBuffer
    (buffer_iter,    CL_TRUE, CL_MAP_READ,  0, int_size_in_bytes);
  float * ptr_error   = (float *) q.enqueueMapBuffer
    (buffer_error,   CL_TRUE, CL_MAP_READ,  0, float_size_in_bytes);

  // Setting input coef_matrix and rhs_vector
  for (size_t i=0; i<N; i++) {
    for (size_t j=0; j<NZ; j++) {
      ptr_ell_col[i*NZ+j] = ell_col[i][j];
      ptr_ell_val[i*NZ+j] = ell_val[i][j];
    }
  }
  for (size_t i=0; i<N; i++) {
    ptr_b[i] = b[i];
  }

  // Transfer data from host to kernel
  q.enqueueMigrateMemObjects
    ({buffer_ell_col, buffer_ell_val, buffer_b},
     0 /* 0 means from host */);

  // Kernel
  q.enqueueTask(krnl);

  /*
  kernel_func
    (ell_col, ell_val,
     b, x,
     max_iter, iter,
     tol_error, ret_error);
  */

  // Transfer data from kernel to host
  q.enqueueMigrateMemObjects
    ({buffer_x, buffer_iter, buffer_error},
     CL_MIGRATE_MEM_OBJECT_HOST);

  // Wait until all tasks in the queue finish
  q.finish();

  // Copy the results
  for (size_t i=0; i<N; i++) {
    x[i] = ptr_x[i];
  }
  iter[0] = ptr_iter[0];
  ret_error[0] = ptr_error[0];

#ifdef USE_TIMESPEC
  clock_gettime(CLOCK_REALTIME, &t2);
#else
  gettimeofday(&t2, NULL);
#endif

  res = calc_res<float>(colptr, col, crs_val, x, b);
  print_stderr_res<float>(iter[0], res);
  print_stderr_error<float>(std::sqrt(ret_error[0]/dot_host<float>(b, b)));
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
