/*--------------------------------------------------------------------
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - CG

  This benchmark is an OpenMP C version of the NPB CG code.
  
  The OpenMP C versions are developed by RWCP and derived from the serial
  Fortran versions in "NPB 2.3-serial" developed by NAS.

  Permission to use, copy, distribute and modify this software for any
  purpose with or without fee is hereby granted.
  This software is provided "as is" without express or implied warranty.
  
  Send comments on the OpenMP C versions to pdp-openmp@rwcp.or.jp

  Information on OpenMP activities at RWCP is available at:

           http://pdplab.trc.rwcp.or.jp/pdperf/Omni/
  
  Information on NAS Parallel Benchmarks 2.3 is available at:
  
           http://www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------*/
/*--------------------------------------------------------------------

  Authors: M. Yarrow
           C. Kuszmaul

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------*/

/*
c---------------------------------------------------------------------
c  Note: please observe that in the routine conj_grad three 
c  implementations of the sparse matrix-vector multiply have
c  been supplied.  The default matrix-vector multiply is not
c  loop unrolled.  The alternate implementations are unrolled
c  to a depth of 2 and unrolled to a depth of 8.  Please
c  experiment with these to find the fastest for your particular
c  architecture.  If reporting timing results, any of these three may
c  be used without penalty.
c---------------------------------------------------------------------
*/
#include "npb-C.h"
#include "npbparams.h"
#include "xmp.h"
#include        <string.h>

#define NUM_PROC_COLS 16
#define NUM_PROC_ROWS 16
#define NUM_PROCS (NUM_PROC_COLS*NUM_PROC_ROWS)
#define	NZ	NA*(NONZER+1)*(NONZER+1)/NUM_PROCS+NA*(NONZER+2+(NUM_PROCS/256))/NUM_PROC_COLS

/* global variables */
int naa, nzz, firstrow, lastrow, firstcol, lastcol;

/* common /main_int_mem/ */
int colidx[NZ];	        /* colidx[0:NZ-1] */
int rowstr[NA+1];	/* rowstr[0:NA] */
int rowstr2[NA+1];
int iv[2*NA+1];	        /* iv[0:2*NA] */
int arow[NZ];		/* arow[0:NZ-1] */
int acol[NZ];		/* acol[0:NZ-1] */

/* common /main_flt_mem/ */
double v[NA+1];	        /* v[0:NA] */
double aelt[NZ];	/* aelt[0:NZ-1] */
double a[NZ];		/* a[0:NZ-1] */
double q[NA], r[NA], p[NA], x[NA], z[NA];

/* common /urando/ */
double amult, tran;

/* function declarations */
void conj_grad(int colidx[], int rowstr[], double a[], double w_shared[NA], 
	       double *rnorm);
void makea(int n, int nz, double a[], int colidx[], int rowstr[],
		  int nonzer, int firstrow, int lastrow, int firstcol,
		  int lastcol, double rcond, int arow[], int acol[],
		  double aelt[], double v[], int iv[], double shift );
void sparse(double a[], int colidx[], int rowstr[], int n,
		   int arow[], int acol[], double aelt[], int firstrow, int lastrow,
		   double x2[], boolean mark[], int nzloc[], int nnza);
void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[]);
int  icnvrt(double x3, int ipwr2);
void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
void setup_submatrix_info();
/*-------------------------------
            for XMP
--------------------------------*/
#pragma xmp nodes pros(16,16)  // pros(NUM_PROC_COLS,NUM_PROC_ROWS)
#pragma xmp template t(0:149999,0:149999) 
double w_shared[NA];    // NUM_ELEMENTS+2*NUM_PROC_COLS
#pragma xmp distribute t(block, block) onto pros
#pragma xmp align w_shared[i] with t(*,i)
#pragma xmp align q[i] with t(i,*)
#pragma xmp align r[i] with t(i,*)
#pragma xmp align p[i] with t(i,*)
#pragma xmp align x[i] with t(i,*)
#pragma xmp align z[i] with t(i,*)
int me, comm_size;

/*--------------------------------------------------------------------
      program cg
--------------------------------------------------------------------*/

int main(int argc, char **argv) {
    int	i, j, k, it, nthreads;
    double zeta, rnorm, norm_temp11, norm_temp12;
    double t, mflops;
    char class;
    boolean verified;
    double zeta_verify_value, epsilon;

    me = xmp_get_rank();
    nthreads = xmp_get_size();

    if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0) {
	class = 'S';
	zeta_verify_value = 8.5971775078648;
    } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0) {
	class = 'W';
	zeta_verify_value = 10.362595087124;
    } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0) {
	class = 'A';
	zeta_verify_value = 17.130235054029;
    } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0) {
	class = 'B';
	zeta_verify_value = 22.712745482631;
    } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0) {
	class = 'C';
	zeta_verify_value = 28.973605592845;
    } else {
	class = 'U';
    }

#pragma xmp task on pros(1,1)
    {
      printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
	     " - CG Benchmark\n");
      printf(" Size: %10d\n", NA);
      printf(" Iterations: %5d\n", NITER);
    }

    naa = NA;
    nzz = NZ;

/*--------------------------------------------------------------------
c  Initialize random number generator
c-------------------------------------------------------------------*/
    tran    = 314159265.0;
    amult   = 1220703125.0;
    zeta    = randlc( &tran, amult );

/* Split the matrix into something smaller */
    setup_submatrix_info();

/*--------------------------------------------------------------------
c  
c-------------------------------------------------------------------*/
    makea(naa, nzz, a, colidx, rowstr, NONZER,
	  firstrow, lastrow, firstcol, lastcol, 
	  RCOND, arow, acol, aelt, v, iv, SHIFT);

/*---------------------------------------------------------------------
c  Note: as a result of the above call to makea:
c        values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
c        values of colidx which are col indexes go from firstcol --> lastcol
c        So:
c        Shift the col index vals from actual (firstcol --> lastcol ) 
c        to local, i.e., (1 --> lastcol-firstcol+1)
c---------------------------------------------------------------------*/
    for (j = 0; j <= lastrow - firstrow+1; j++)
      rowstr2[j+firstrow-1] = rowstr[j];
    
    for (j = 0; j <= lastrow - firstrow+1; j++)
      rowstr[j+firstrow-1] = rowstr2[j+firstrow-1];

/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c-------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
    for (j = 0; j < NA; j++)
      x[j] = 1.0;

    zeta  = 0.0;
/*-------------------------------------------------------------------
c---->
c  Do one iteration untimed to init all code and data page tables
c---->                    (then reinit, start timing, to niter its)
c-------------------------------------------------------------------*/

/*--------------------------------------------------------------------
c  The call to the conjugate gradient routine:
c-------------------------------------------------------------------*/
    conj_grad(colidx, rowstr, a,  w_shared, &rnorm);

/*--------------------------------------------------------------------
c  zeta = shift + 1/(x.z)
c  So, first: (x.z)
c  Also, find norm of z
c  So, first: (z.z)
c-------------------------------------------------------------------*/
    norm_temp11 = 0.0;
    norm_temp12 = 0.0;

#pragma xmp loop on t(i,*)
    for (i = 0; i < NA; i++ ){
      norm_temp11 = norm_temp11 + x[j]*z[j];
      norm_temp12 = norm_temp12 + z[j]*z[j];
    }

#pragma xmp reduction(+:norm_temp11) on pros(:,*)
#pragma xmp reduction(+:norm_temp12) on pros(:,*)

    norm_temp12 = 1.0 / sqrt( norm_temp12 );

/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
#pragma xmp loop on t(i,*)
    for (i = 0; i < NA; i++ )
      x[j] = norm_temp12*z[j];

/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c-------------------------------------------------------------------*/
#pragma xmp loop on t(i,*)
    for (i = 0; i < NA; i++ )
      x[i] = 1.0;

    zeta  = 0.0;

    timer_clear( 1 );

#pragma xmp barrier
    timer_start( 1 );
    
/*--------------------------------------------------------------------
c---->
c  Main Iteration for inverse power method
c---->
c-------------------------------------------------------------------*/

    for (it = 1; it <= NITER; it++) {
/*--------------------------------------------------------------------
c  The call to the conjugate gradient routine:
c-------------------------------------------------------------------*/
      conj_grad(colidx, rowstr, a, w_shared, &rnorm);

/*--------------------------------------------------------------------
c  zeta = shift + 1/(x.z)
c  So, first: (x.z)
c  Also, find norm of z
c  So, first: (z.z)
c-------------------------------------------------------------------*/
	norm_temp11 = 0.0;
	norm_temp12 = 0.0;

#pragma xmp loop on t(j,*)
	for (j = 0; j < NA; j++) {
	  norm_temp11 = norm_temp11 + x[j] * z[j];
	  norm_temp12 = norm_temp12 + z[j] * z[j];
	}

#pragma xmp reduction(+:norm_temp11) on pros(:,*)
#pragma xmp reduction(+:norm_temp12) on pros(:,*)
	
	norm_temp12 = 1.0 / sqrt( norm_temp12 );
	zeta = SHIFT + 1.0 / norm_temp11;

	if(me == 0){   // "If statement" may be faster than "task directive".
	  if( it == 1 )
	    printf("   iteration           ||r||                 zeta\n");
	
	  printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);
	} 
/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
	for (j = 0; j < NA; j++)
	  x[j] = norm_temp12*z[j];
    
    } /* end of main iter inv pow meth */

    timer_stop( 1 );

/*--------------------------------------------------------------------
c  End of timed section
c-------------------------------------------------------------------*/
    t = timer_read( 1 );

#pragma xmp reduction(MAX:t)
#pragma xmp task on pros(1,1)
    {
    printf(" Benchmark completed\n");

    epsilon = 1.0e-10;
    if (class != 'U') {
	if (fabs(zeta - zeta_verify_value) <= epsilon) {
            verified = TRUE;
	    printf(" VERIFICATION SUCCESSFUL\n");
	    printf(" Zeta is    %20.12e\n", zeta);
	    printf(" Error is   %20.12e\n", zeta - zeta_verify_value);
	} else {
            verified = FALSE;
	    printf(" VERIFICATION FAILED\n");
	    printf(" Zeta                %20.12e\n", zeta);
	    printf(" The correct zeta is %20.12e\n", zeta_verify_value);
	}
    } else {
	verified = FALSE;
	printf(" Problem size unknown\n");
	printf(" NO VERIFICATION PERFORMED\n");
    }

    if ( t != 0.0 ) {
	mflops = (2.0*NITER*NA)
	    * (3.0+(NONZER*(NONZER+1)) + 25.0*(5.0+(NONZER*(NONZER+1))) + 3.0 )
	    / t / 1000000.0;
    } else {
	mflops = 0.0;
    }

    c_print_results("CG", class, NA, 0, 0, NITER, nthreads, t, 
		    mflops, "          floating point", 
		    verified, NPBVERSION, COMPILETIME,
		    CS1, CS2, CS3, CS4, CS5, CS6, CS7);
    }
    return 0;
}

void setup_submatrix_info( void ){
  int col_size, row_size;
  int t, i, j;
  int npcols = NUM_PROC_COLS;
  int nprows = NUM_PROC_ROWS;
  int proc_row = me / npcols;
  int proc_col = me - (proc_row*npcols);

  if( (naa/npcols*npcols) == naa ){
    col_size = naa/npcols;
    firstcol = proc_col*col_size + 1;
    lastcol = firstcol - 1 + col_size;
    row_size = naa / nprows;
    firstrow = proc_row*row_size + 1;
    lastrow = firstrow - 1 + row_size;
  }
  else{
    if( proc_row < (naa - (naa / (nprows*nprows))) ){
      row_size = (naa / nprows); // + 1;                       
      firstrow = (proc_row * row_size) + 1;
      lastrow = firstrow - 1 + row_size;
    }
    else {
      row_size = naa / nprows;
      firstrow = (naa - (naa/(nprows*nprows))) * (row_size + 1)
	+ (proc_row - (naa - (naa/(nprows * nprows))) * row_size) + 1;
      lastrow = firstrow - 1 + row_size;
    }
    if( npcols == nprows ) {
      if( proc_col < (naa - (naa / (npcols*npcols))) ) {
	col_size = (naa / npcols); // + 1;
	firstcol = (proc_col * col_size) + 1;
	lastcol = firstcol - 1 + col_size;
      }
      else {
	col_size = naa / npcols;
	firstcol = (naa - (naa/(npcols*npcols))) * (col_size + 1)
	  + (proc_col - (naa - (naa/(npcols * npcols))) * col_size) + 1;
	lastcol = firstcol - 1 + col_size;
      }
    } 
    else {
      if( (proc_col/2) < (naa - (naa / ((npcols/2)*(npcols/2)))) ) {
	col_size = naa / (npcols/2); // + 1;
	firstcol = (proc_col/2) * col_size + 1;
	lastcol = firstcol - 1 + col_size;
      }
      else {
	col_size = naa / (npcols/2);
	firstcol = (naa - (naa/((npcols/2)*(npcols/2)))) * (col_size + 1)
		     + ((proc_col/2) - (naa - (naa/((npcols/2) * (npcols/2)))) * col_size) + 1;
	lastcol = firstcol - 1 + col_size;
      }
      if( (t%2) == 0 )
	lastcol = firstcol - 1 + (col_size-1)/2 + 1;
      else {
	firstcol = firstcol + (col_size-1)/2 + 1;
	lastcol = firstcol - 1 + col_size/2;
      }
    }
  }
}


/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/
void conj_grad (
    int colidx[],	/* colidx[1:nzz] */
    int rowstr[],	/* rowstr[1:naa+1] */
    double a[],		/* a[1:nzz] */
    double w_shared[NA],
    double *rnorm )
/*---------------------------------------------------------------------
c  Floaging point arrays here are named as in NPB1 spec discussion of 
c  CG algorithm
c---------------------------------------------------------------------*/
{
#pragma xmp barrier

    static double d, sum, rho, rho0, alpha, beta;
    int i, j, k, yy;
    int cgit, cgitmax = 25;
    rho = 0.0;

/*--------------------------------------------------------------------
c  Initialize the CG algorithm:
c-------------------------------------------------------------------*/
    memset( &z[firstcol-1], 0, sizeof(double) * (naa/NUM_PROC_COLS));
    memset( &q[firstcol-1], 0, sizeof(double) * (naa/NUM_PROC_COLS)); 
    memcpy( &r[firstcol-1], &x[firstcol-1], sizeof(double) * (naa/NUM_PROC_COLS));  
    memcpy( &p[firstcol-1], &x[firstcol-1], sizeof(double) * (naa/NUM_PROC_COLS));
    memset( &w_shared[firstrow-1], 0, sizeof(double) * (naa/NUM_PROC_ROWS));

/*--------------------------------------------------------------------
c  rho = r.r
c  Now, obtain the norm of r: First, sum squares of r elements locally...
c-------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
    for (j = 0; j < NA; j++)
      rho = rho + x[j]*x[j];

#pragma xmp reduction(+:rho) on pros(:,*)

/*--------------------------------------------------------------------
c---->
c  The conj grad iteration loop
c---->
c-------------------------------------------------------------------*/
    for (cgit = 1; cgit <= cgitmax; cgit++) {
      rho0 = rho;
      d = 0.0;
      rho = 0.0;
      
/*--------------------------------------------------------------------
c  q = A.p
c  The partition submatrix-vector multiply: use workspace w
c---------------------------------------------------------------------
C
C  NOTE: this version of the multiply is actually (slightly: maybe %5) 
C        faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
C        below.   On the Cray t3d, the reverse is true, i.e., the 
C        unrolled-by-two version is some 10% faster.  
C        The unrolled-by-8 version below is significantly faster
C        on the Cray t3d - overall speed of code is 1.5 times faster.
*/

/* rolled version */      
#pragma xmp loop on t(*,j)
      for(j=0; j < NA; j++){
	sum = 0.0;
	for (k = rowstr[j]; k < rowstr[j+1]; k++) {
	  sum = sum + a[k-1]*p[colidx[k-1]-1];
	}
	w_shared[j] = sum;
      }

#pragma xmp reduction(+:w_shared) on pros(:,*)

#pragma xmp gmove
      q[:] = w_shared[:];

/*--------------------------------------------------------------------
c  Clear w for reuse...
c-------------------------------------------------------------------*/

/*--------------------------------------------------------------------
c  Obtain p.q
c-------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
	for (j = 0; j < NA; j++)
	  d = d + p[j] * q[j];

#pragma xmp reduction(+:d) on pros(:,*)

/*--------------------------------------------------------------------
c  Obtain alpha = rho / (p.q)
c-------------------------------------------------------------------*/
	alpha = rho0 / d;

/*--------------------------------------------------------------------
c  Save a temporary of rho
c-------------------------------------------------------------------*/
	/*	rho0 = rho;*/

/*---------------------------------------------------------------------
c  Obtain z = z + alpha*p
c  and    r = r - alpha*q
c---------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
	for (j = 0; j < NA; j++ ){
	  z[j] = z[j] + alpha*p[j];
	  r[j] = r[j] - alpha*q[j];
	}
            
/*---------------------------------------------------------------------
c  rho = r.r
c  Now, obtain the norm of r: First, sum squares of r elements locally...
c---------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
        for (j = 0; j < NA; j++ )
	  rho = rho + r[j] * r[j];

#pragma xmp reduction(+:rho) on pros(:,*)

/*--------------------------------------------------------------------
c  Obtain beta:
c-------------------------------------------------------------------*/
	beta = rho / rho0;

/*--------------------------------------------------------------------
c  p = r + beta*p
c-------------------------------------------------------------------*/
#pragma xmp loop on t(j,*)
	for (j = 0; j < NA; j++)
	  p[j] = r[j] + beta*p[j];

    } /* end of do cgit=1,cgitmax */

/*---------------------------------------------------------------------
c  Compute residual norm explicitly:  ||r|| = ||x - A.z||
c  First, form A.z
c  The partition submatrix-vector multiply
c---------------------------------------------------------------------*/
#pragma xmp loop on t(*,j)
    for(j=0; j < NA; j++){
	d = 0.0;
	for (k = rowstr[j]-1; k < rowstr[j+1]-1; k++) {
	  d = d + a[k]*z[colidx[k]-1];
	}
	w_shared[j] = d;
    }

#pragma xmp reduction(+:w_shared) on pros(:,*)

#pragma xmp gmove
    r[:] = w_shared[:];

/*--------------------------------------------------------------------
c  At this point, r contains A.z
c-------------------------------------------------------------------*/
    sum = 0.0;
#pragma xmp loop on t(j,*)
    for (j = 0; j < NA; j++){
	d = x[j] - r[j];
	sum = sum + d*d;
    }

#pragma xmp reduction(+:sum) on pros(:,*)

    (*rnorm) = sqrt(sum);
}

/*---------------------------------------------------------------------
c       generate the test problem for benchmark 6
c       makea generates a sparse matrix with a
c       prescribed sparsity distribution
c
c       parameter    type        usage
c
c       input
c
c       n            i           number of cols/rows of matrix
c       nz           i           nonzeros as declared array size
c       rcond        r*8         condition number
c       shift        r*8         main diagonal shift
c
c       output
c
c       a            r*8         array for nonzeros
c       colidx       i           col indices
c       rowstr       i           row pointers
c
c       workspace
c
c       iv, arow, acol i
c       v, aelt        r*8
c---------------------------------------------------------------------*/
void makea(
    int n,
    int nz,
    double a[],		/* a[0:nz-1] */
    int colidx[],	/* colidx[0:nz-1] */
    int rowstr[],	/* rowstr[0:n] */
    int nonzer,
    int firstrow,
    int lastrow,
    int firstcol,
    int lastcol,
    double rcond,
    int arow[],		/* arow[0:nz-1] */
    int acol[],		/* acol[0:nz-1] */
    double aelt[],	/* aelt[0:nz-1] */
    double v[],		/* v[0:n] */
    int iv[],		/* iv[0:2*n] */
    double shift )
{
    int i, nnza, iouter, ivelt, ivelt1, irow, nzv;

/*--------------------------------------------------------------------
c      nonzer is approximately  (int(sqrt(nnza /n)));
c-------------------------------------------------------------------*/

    double size, ratio, scale;
    int jcol;

    size = 1.0;
    ratio = pow(rcond, (1.0 / (double)n));
    nnza = 0;

/*---------------------------------------------------------------------
c  Initialize colidx(n+1 .. 2n) to zero.
c  Used by sprnvc to mark nonzero positions
c---------------------------------------------------------------------*/
    for (i = 1; i <= n; i++) 
	colidx[n+i-1] = 0;

    for (iouter = 1; iouter <= n; iouter++) {
	nzv = nonzer;
	sprnvc(n, nzv, v, iv, &(colidx[0]), &(colidx[n]));
	vecset(n, v, iv, &nzv, iouter, 0.5);
	for (ivelt = 1; ivelt <= nzv; ivelt++) {
	    jcol = iv[ivelt-1];
	    if (jcol >= firstcol && jcol <= lastcol) {
		scale = size * v[ivelt-1];
		for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
	            irow = iv[ivelt1-1];
                    if (irow >= firstrow && irow <= lastrow) {
			nnza = nnza + 1;
			if (nnza > nz) {
			    printf("Space for matrix elements exceeded in makea\n");
			    printf("nnza, nzmax = %d, %d\n", nnza, nz);
			    printf("iouter = %d\n", iouter);
			    exit(1);
			}
			acol[nnza-1] = jcol;
			arow[nnza-1] = irow;
			aelt[nnza-1] = v[ivelt1-1] * scale;
		    }
		}
	    }
	}
	size = size * ratio;
    }

/*---------------------------------------------------------------------
c       ... add the identity * rcond to the generated matrix to bound
c           the smallest eigenvalue from below by rcond
c---------------------------------------------------------------------*/
    for (i = firstrow; i <= lastrow; i++) {
	if (i >= firstcol && i <= lastcol) {
	    iouter = n + i;
	    nnza = nnza + 1;
	    if (nnza > nz) {
		printf("Space for matrix elements exceeded in makea\n");
		printf("nnza, nzmax = %d, %d\n", nnza, nz);
		printf("iouter = %d\n", iouter);
		exit(1);
	    }
	    acol[nnza-1] = i;
	    arow[nnza-1] = i;
	    aelt[nnza-1] = rcond - shift;
	}
    }

/*---------------------------------------------------------------------
c       ... make the sparse matrix from list of elements with duplicates
c           (v and iv are used as  workspace)
c---------------------------------------------------------------------*/
    sparse(a, colidx, rowstr, n, arow, acol, aelt,
	   firstrow, lastrow, v, &(iv[0]), &(iv[n]), nnza);
}

/*---------------------------------------------------
c       generate a sparse matrix from a list of
c       [col, row, element] tri
c---------------------------------------------------*/
void sparse(
    double a[],		/* a[0:*] */
    int colidx[],	/* colidx[0:*] */
    int rowstr[],	/* rowstr[0:*] */
    int n,
    int arow[],		/* arow[0:*] */
    int acol[],		/* acol[0:*] */
    double aelt[],	/* aelt[0:*] */
    int firstrow,
    int lastrow,
    double x2[],		/* x[0:n-1] */
    boolean mark[],	/* mark[0:n-1] */
    int nzloc[],	/* nzloc[0:n-1] */
    int nnza)
/*---------------------------------------------------------------------
c       rows range from firstrow to lastrow
c       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
c---------------------------------------------------------------------*/
{
    int nrows;
    int i, j, jajp1, nza, k, nzrow;
    double xi;

/*--------------------------------------------------------------------
c    how many rows of result
c-------------------------------------------------------------------*/
    nrows = lastrow - firstrow + 1;

/*--------------------------------------------------------------------
c     ...count the number of triples in each row
c-------------------------------------------------------------------*/
    for (j = 1; j <= n; j++) {
	rowstr[j-1] = 0;
	mark[j-1] = FALSE;
    }
    rowstr[n] = 0;
    
    for (nza = 1; nza <= nnza; nza++) {
	j = (arow[nza-1] - firstrow + 1) + 1;
	rowstr[j-1] = rowstr[j-1] + 1;
    }

    rowstr[0] = 1;
    for (j = 2; j <= nrows+1; j++)
	rowstr[j-1] = rowstr[j-1] + rowstr[j-2];

/*---------------------------------------------------------------------
c     ... rowstr(j) now is the location of the first nonzero
c           of row j of a
c---------------------------------------------------------------------*/
    
/*--------------------------------------------------------------------
c     ... do a bucket sort of the triples on the row index
c-------------------------------------------------------------------*/
    for (nza = 1; nza <= nnza; nza++) {
	j = arow[nza-1] - firstrow + 1;
	k = rowstr[j-1];
	a[k-1] = aelt[nza-1];
	colidx[k-1] = acol[nza-1];
	rowstr[j-1] = rowstr[j-1] + 1;
    }

/*--------------------------------------------------------------------
c       ... rowstr(j) now points to the first element of row j+1
c-------------------------------------------------------------------*/
    for (j = nrows; j >= 1; j--) {
	rowstr[j] = rowstr[j-1];
    }
    rowstr[0] = 1;

/*--------------------------------------------------------------------
c       ... generate the actual output rows by adding elements
c-------------------------------------------------------------------*/
    nza = 0;
    for (i = 0; i < n; i++) {
      	x2[i] = 0.0;
	mark[i] = FALSE;
    }

    jajp1 = rowstr[0];
    for (j = 1; j <= nrows; j++) {
	nzrow = 0;
/*--------------------------------------------------------------------
c          ...loop over the jth row of a
c-------------------------------------------------------------------*/
	for (k = jajp1; k < rowstr[j]; k++) {
            i = colidx[k-1];
	    x2[i-1] = x2[i-1] + a[k-1];
	    if ( mark[i-1] == FALSE && x2[i-1] != 0.0) {
		mark[i-1] = TRUE;
		nzrow = nzrow + 1;
		nzloc[nzrow-1] = i;
	    }
	}

/*--------------------------------------------------------------------
c          ... extract the nonzeros of this row
c-------------------------------------------------------------------*/
	for (k = 1; k <= nzrow; k++) {
            i = nzloc[k-1];
            mark[i-1] = FALSE;
	    xi = x2[i-1];
	    x2[i-1] = 0.0;
            if (xi != 0.0) {
		nza = nza + 1;
		a[nza-1] = xi;
		colidx[nza-1] = i;
	    }
	}
	jajp1 = rowstr[j];
	rowstr[j] = nza + rowstr[0];
    }
}

/*---------------------------------------------------------------------
c       generate a sparse n-vector (v, iv)
c       having nzv nonzeros
c
c       mark(i) is set to 1 if position i is nonzero.
c       mark is all zero on entry and is reset to all zero before exit
c       this corrects a performance bug found by John G. Lewis, caused by
c       reinitialization of mark on every one of the n calls to sprnvc
---------------------------------------------------------------------*/
void sprnvc(
    int n,
    int nz,
    double v[],		/* v[0:*] */
    int iv[],		/* iv[0:*] */
    int nzloc[],	/* nzloc[0:n-1] */
    int mark[] ) 	/* mark[0:n-1] */
{
    int nn1;
    int nzrow, nzv, ii, i;
    double vecelt, vecloc;

    nzv = 0;
    nzrow = 0;
    nn1 = 1;
    do {
	nn1 = 2 * nn1;
    } while (nn1 < n);

/*--------------------------------------------------------------------
c    nn1 is the smallest power of two not less than n
c-------------------------------------------------------------------*/
    while (nzv < nz) {
	vecelt = randlc(&tran, amult);
/*--------------------------------------------------------------------
c   generate an integer between 1 and n in a portable manner
c-------------------------------------------------------------------*/
	vecloc = randlc(&tran, amult);
	i = icnvrt(vecloc, nn1) + 1;
	if (i > n) continue;

/*--------------------------------------------------------------------
c  was this integer generated already?
c-------------------------------------------------------------------*/
	if (mark[i-1] == 0) {
	    mark[i-1] = 1;
	    nzrow = nzrow + 1;
	    nzloc[nzrow-1] = i;
	    nzv = nzv + 1;
	    v[nzv-1] = vecelt;
	    iv[nzv-1] = i;
	}
    }

    for (ii = 1; ii <= nzrow; ii++) {
	i = nzloc[ii-1];
	mark[i-1] = 0;
    }
}

/*---------------------------------------------------------------------
* scale a double precision number x in (0,1) by a power of 2 and chop it
*---------------------------------------------------------------------*/
int icnvrt(double x3, int ipwr2) {
    return ((int)(ipwr2 * x3));
}

/*--------------------------------------------------------------------
c       set ith element of sparse vector (v, iv) with
c       nzv nonzeros to val
c-------------------------------------------------------------------*/
void vecset(
    int n,
    double v[],	/* v[0:*] */
    int iv[],	/* iv[0:*] */
    int *nzv,
    int i,
    double val)
{
    int k;
    boolean set;

    set = FALSE;
    for (k = 1; k <= *nzv; k++) {
	if (iv[k-1] == i) {
            v[k-1] = val;
            set  = TRUE;
	}
    }
    if (set == FALSE) {
	*nzv = *nzv + 1;
	v[*nzv-1] = val;
	iv[*nzv-1] = i;
    }
}
