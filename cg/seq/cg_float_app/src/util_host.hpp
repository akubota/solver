#ifndef __INCLUDED_UTIL_HOST_HPP__
#define __INCLUDED_UTIL_HOST_HPP__

#include "sparse_defs.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>    /* std::sqrt */
#include <unistd.h> /* getopt */
#include <stdlib.h>

#undef DEBUG

/**********************************************************************/
/* Matrix Market, CCS, CRS and ELL formats */
/**********************************************************************/
int readHeaders(std::ifstream & infile,
		int & rows, int & cols, int & nnzs, bool & issymmetric)
{
  std::string buf;

  // 1st line
  infile >> buf;      // %%MatrixMarket
  infile >> buf;      // matrix

  infile >> buf;      // coordinate
  infile >> buf;
  infile >> buf;
  if (buf == "symmetric") {
    issymmetric = true;
  } else {
    issymmetric = false;
  }

  // rows, cols, nnzs
  infile >> rows >> cols >> nnzs;
  return 0;
}

// Warning: assume non-zero values and row indecies are sorted
template <typename T>
int mmtoccs(std::ifstream & infile, int * rowptr, int * row, T * val, T * diag,
	    int nnzs, int cols)
{
  // load non-zero values

  for (int i=0; i <= cols; i++) {
    rowptr[i] = 0;
  }

  int curr_col = 1;
  for (int el=0; el<nnzs; el++) {
    int r, col;
    infile >> r >> col >> val[el];
    if (r == col) {
      diag[r-1] = val[el];
    }
    row[el] = r-1;
    if (col != curr_col) {
      for (int c=curr_col+1; c<=col; c++) {
	rowptr[c] = rowptr[curr_col];
      }
      curr_col = col;
    }
    rowptr[col]++;
  }
  for (int c=curr_col+1; c<=cols; c++) {
    rowptr[c] = rowptr[curr_col];
  }

  return 0;
}

template <typename T>
void
ccstocrs(int * colptr, int * col, T * crs_val,
	 int * rowptr, int * row, T * val,
	 //	 int nnzs,
	 int rows, int cols)
{
  int i, j;
  int rb, re;
  int r;

  int * colsInRow = new int[rows];

  for (r=0; r<rows; r++) {
    colsInRow[r] = 0;
  }

  for (i=0; i<cols; i++) {
    rb = rowptr[i];
    re = rowptr[i+1];
    for (j=rb; j<re; j++) {
      //  mat(row[j], i) : val[j]
      colsInRow[row[j]]++;
    }
  }

  colptr[0] = 0;
  for (r=1; r<=rows; r++) {
    colptr[r] = colptr[r-1] + colsInRow[r-1];
  }

  for (r=0; r<rows; r++) {
    colsInRow[r] = 0;
  }

  for (i=0; i<cols; i++) {
    rb = rowptr[i];
    re = rowptr[i+1];
    for (j=rb; j<re; j++) {
      // mat(row[j], i) : val[j];
      // -> mat(row[j], col[i]) : crs_val[i]
      r = row[j];
      col[colptr[r]+colsInRow[r]] = i;
      crs_val[colptr[r]+colsInRow[r]] = val[j];
      colsInRow[r]++;
    }
  }

  delete[] colsInRow;
}

template <typename T>
void
symccstocrs(int * colptr, int * col, T * crs_val,
	    int * rowptr, int * row, T * val,
	    //int nnzs,
	    int rows, int cols)
{
  int i, j;
  int rb, re;
  int r;

  int * colsInRow = new int[rows];

  for (r=0; r<rows; r++) {
    colsInRow[r] = 0;
  }
  
  for (i=0; i<cols; i++) {
    rb = rowptr[i];
    re = rowptr[i+1];
    for (j=rb; j<re; j++) {
      //  mat(row[j], i) : val[j]
      colsInRow[row[j]]++;
      if (row[j] != i) {
	colsInRow[i]++;
      }
    }
  }

  colptr[0] = 0;
  for (r=1; r<=rows; r++) {
    colptr[r] = colptr[r-1] + colsInRow[r-1];
  }

  for (r=0; r<rows; r++) {
    colsInRow[r] = 0;
  }

  for (i=0; i<cols; i++) {
    rb = rowptr[i];
    re = rowptr[i+1];
    for (j=rb; j<re; j++) {
      // mat(row[j], i) : val[j];
      // -> mat(row[j], col[i]) : crs_val[i]
      r = row[j];
      col[colptr[r]+colsInRow[r]] = i;
      crs_val[colptr[r]+colsInRow[r]] = val[j];
      colsInRow[r]++;

      if (row[j] != i) {
	col[colptr[i]+colsInRow[i]] = r;
	crs_val[colptr[i]+colsInRow[i]] = val[j];
	colsInRow[i]++;
      }
    }
  }

  delete[] colsInRow;
}

int getMaxNZ(int * colptr, int * col)
{
  int maxNZ = 0;
  int tmpNZ;

  for (int i=0; i<N; i++) {
    tmpNZ = colptr[i+1]-colptr[i];
    if (tmpNZ > maxNZ) {
      maxNZ = tmpNZ;
    }
  }
  return maxNZ;
}

template<typename T>
void crstoellcrs
(int * ell_col, T * ell_val,
 int * colptr, int * col, T * val,
 int rows, int max_nz)
{
  int i, j, k;

  for (i=0; i<rows; i++) {
    int cb, ce;
    cb = colptr[i];
    ce = colptr[i+1];
    for (j=cb, k=0; j<ce; j++, k++) {
      ell_col[i*max_nz + k] = col[j];
      ell_val[i*max_nz + k] = val[j];
    }

    // fill dummy values in ell_col and ell_val
    if (ce-cb>0) {
      for (k=ce-cb; k<max_nz; k++) {
	ell_col[i*max_nz + k] = col[ce-1];
	ell_val[i*max_nz + k] = (T)0;
      }
    } else if (cb-1>=0) {
      for (k=0; k<max_nz; k++) {
	ell_col[i*max_nz + k] = col[cb-1];
	ell_val[i*max_nz + k] = (T)0;
      }
    } else {
      for (k=0; k<max_nz; k++) {
	ell_col[i*max_nz + k] = 0;
	ell_val[i*max_nz + k] = (T)0;
      }
    }
  }

  return;
}

/**********************************************************************/
/* parseArgsGetOpt */
/**********************************************************************/

template<typename T>
void parseArgsGetOpt
(int argc, char **argv,
 int &max_iter, T &tol_error,
 const char ** coef_filename_ptr)
{
  int opt;
  opterr = 0;

  while((opt = getopt(argc, argv, "i:r:m:")) != -1) {
    switch (opt) {
    case 'i':
      max_iter = strtol(optarg, NULL, 10);
      break;
    case 'r':
      tol_error = strtod(optarg, NULL);
      break;
    case 'm':
      *coef_filename_ptr = optarg;
      break;
    default:
      ;
    }
  }
  return;
}

/**********************************************************************/
/* print */
/**********************************************************************/

template <typename T>
void
print_vec(T v[N])
{
  int i;
  for (i=0; i<N; i++) {
    std::cout << i << ": " << v[i] << std::endl;
  }
  return;
}

template <typename T>
void
print_res(int iter, T res)
{
  std::cout << "res:" << iter << " = " << res << std::endl;
}

template <typename T>
void
print_stderr_res(int iter, T res)
{
  std::cerr << "res:" << iter << " = " << res << std::endl;
}

template <typename T>
void
print_stderr_error(T error)
{
  std::cerr << "error = " << error << std::endl;
}

/**********************************************************************/
/* res */
/**********************************************************************/

template <typename T>
T
calc_res
(int colptr[N+1],
 int col[NV],
 T val[NV],
 T x[N],
 T b[N]
 )
{
  int i, j;
  int cb, ce;
  T tmp;
  T ret = (T)0;

  for (i=0; i<N; i++) {
    cb = colptr[i];
    ce = colptr[i+1];
    tmp = (T)0;
    for (j=cb; j<ce; j++) {
      tmp += val[j] * x[col[j]];
    }
    ret = ret + (tmp - b[i]) * (tmp - b[i]);
  }
  ret = std::sqrt(ret);
  return ret;
}

/**********************************************************************/
/* init_rhs */
/**********************************************************************/

template <typename T>
void
init_rhs
(int colptr[N+1],
 int col[NV],
 T val[NV],
 T x[N],
 T b[N]
 )
{
  int i, j;
  int cb, ce;
  T tmp;

  // set temporary solution
  for (i=0; i<N; i++) {
    x[i] = (T)1;
  }

  // set rhs vector
  for (i=0; i<N; i++) {
    cb = colptr[i];
    ce = colptr[i+1];
    tmp = (T)0;
    for (j=cb; j<ce; j++) {
      tmp += val[j] * x[col[j]];
    }
    b[i] = tmp;
  }

  // clear temporary solution
  for (i=0; i<N; i++) {
    x[i] = (T)0;
  }
  return;
}

#endif
