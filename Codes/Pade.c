// gcc Fox_Li.c -lm -lgsl -lgslcblas -o Fox_Li
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_gamma.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

void hns_gen(int N, double * hns);
// _Complex double * Kung(int N, double * hns);
void Kung(int N, double * hns);
gsl_matrix * diagonalise(gsl_vector * v, int M, int N);
gsl_vector * vector_truncate(gsl_vector * v, int M);
gsl_matrix * matrix_extract(gsl_matrix * A, int M_start, int M_end, int N_start, int N_end);
gsl_matrix * construct_SM_sqrt(gsl_vector * s, int M);
gsl_matrix * construct_Hankel(double * hns, int N);
gsl_matrix * construct_A(gsl_matrix * U_tilde, gsl_matrix * SM_sqrt, gsl_matrix * U_hat, int M, int N);
gsl_vector * construct_b(gsl_matrix * SM_sqrt, gsl_matrix * V, int M);
gsl_vector * construct_c(gsl_matrix * U, gsl_matrix * SM_sqrt, int M);
gsl_matrix * inv_diagonalise(gsl_vector * v, int M, int N);

int main(void) {

	/* Generate the ratio of the Maclaurin coefficients */
	/* ------------------------------------------------ */
	int N = 30;	// Number of coefficients // No point in going any bigger
	double * hns = (double *) malloc((2*N + 1) * sizeof(double));
	hns_gen(N, hns);

	/* Solve the relaxed moment problem for sinc and Gaussian */
	/* ------------------------------------------------------ */
	Kung(N, hns);

	return 0;
}

void hns_gen(int N, double * hns) {
	for (int n = 0; n < 2*N + 1; n++)
		hns[n] = gsl_sf_gamma(0.5*(n + 2)) / gsl_sf_fact(n + 1);
}

void Kung(int N, double * hns) {

	int M = 1;
	double eps_sq = 1e-16;
	gsl_matrix * HN = gsl_matrix_alloc(N + 1, N + 1);
	gsl_vector * s = gsl_vector_alloc(N + 1);
	gsl_vector * work = gsl_vector_alloc(N + 1);
	gsl_matrix * V = gsl_matrix_alloc(N + 1, N + 1);

	// Constuct the Hankel matrix HN
	HN = construct_Hankel(hns, N);

	// Perform an SVD on HN
	gsl_linalg_SV_decomp(HN, V, s, work);

	// Find truncation parameter M for the given tolerance
	while (sqrt(M)*gsl_vector_get(s, M + 1) >= eps_sq)
		M++;

	// Contruct the M-truncated singular value matrix
	gsl_matrix * SM_sqrt = gsl_matrix_alloc(M, M);
	SM_sqrt = construct_SM_sqrt(s, M);

	// Construct the other matrices for the realisation triple operations
	gsl_matrix * U_tilde = gsl_matrix_alloc(N - 1, M);
	gsl_matrix * U_hat = gsl_matrix_alloc(N - 1, M);
	U_tilde = matrix_extract(HN, 0, N - 2, 0, M - 1);
	U_hat = matrix_extract(HN, 1, N - 1, 0, M - 1);

	// Form the realisation triple (A, b, c)
	gsl_matrix * A = gsl_matrix_alloc(M, M);
	gsl_vector * b = gsl_vector_alloc(M);
	gsl_vector * c = gsl_vector_alloc(M);
	A = construct_A(U_tilde, SM_sqrt, U_hat, M, N);
	b = construct_b(SM_sqrt, V, M); // Note V is not transpose
	c = construct_c(HN, SM_sqrt, M);

	for (int m = 0; m < N - 1; m++) {
		for (int n = 0; n < M; n++)
			printf("%.15lf\n", gsl_matrix_get(U_hat, m, n));
	}


}

gsl_vector * construct_b(gsl_matrix * SM_sqrt, gsl_matrix * V, int M) {
	double bm;
	gsl_vector * b = gsl_vector_alloc(M);
	gsl_matrix * b_temp = gsl_matrix_alloc(M, 1);
	gsl_matrix * Vstar_vec = gsl_matrix_alloc(1, M);
	Vstar_vec = matrix_extract(V, 0, 0, 0, M - 1);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, SM_sqrt, Vstar_vec, 0.0, b_temp);
	for (int m = 0; m < M; m++) {
		bm = gsl_matrix_get(b_temp, m, 0);
		gsl_vector_set(b, m, bm);
	}
	return b;
}

gsl_vector * construct_c(gsl_matrix * U, gsl_matrix * SM_sqrt, int M) {
	double cm;
	gsl_vector * c = gsl_vector_alloc(M);
	gsl_matrix * c_temp = gsl_matrix_alloc(1, M);
	gsl_matrix * U_row = gsl_matrix_alloc(1, M);
	U_row = matrix_extract(U, 0, 0, 0, M - 1);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U_row, SM_sqrt, 0.0, c_temp);
	for (int m = 0; m < M; m++) {
		cm = gsl_matrix_get(c_temp, 0, m);
		gsl_vector_set(c, m, cm);
	}
	return c;
}

gsl_matrix * construct_A(gsl_matrix * U_tilde, gsl_matrix * SM_sqrt, gsl_matrix * U_hat, int M, int N) {
	// dim(U_tilde x SM_sqrt = (N - 1 x M))
	// dim(U_hat x SM_sqrt = (N - 1 x M))
	// dim(A) = M x M
	double sm;
	gsl_matrix * U_tildeS = gsl_matrix_alloc(N - 1, M);
	gsl_matrix * U_tildeSinv = gsl_matrix_alloc(M, N - 1);
	gsl_matrix * U_hatS = gsl_matrix_alloc(N - 1, M);
	gsl_matrix * V = gsl_matrix_alloc(M, M);
	gsl_matrix * A = gsl_matrix_alloc(M, M);
	gsl_vector * s = gsl_vector_alloc(M);
	gsl_matrix * Sinv = gsl_matrix_alloc(N - 1, M);
	gsl_vector * work = gsl_vector_alloc(M);
	gsl_matrix * VSinv = gsl_matrix_alloc(M, N - 1);

	// Compute U_tilde * SM_sqrt
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U_tilde, SM_sqrt, 0.0, U_tildeS);

	// Find inverse of U_tildeS
	// SVD U_tildeS and find inverse of singular value matrix
	gsl_linalg_SV_decomp(U_tildeS, V, s, work);
	Sinv = inv_diagonalise(s, N - 1, M);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, Sinv, 0.0, VSinv);
	// Transpose U_tildeS = U and do VSinv * U_tildeS to find pinv(U_tildeS)
	my_transpose(U_tildeS);


	// gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U_hat, SM_sqrt, 0.0, U_hatS);
	// gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U_tildeS, U_hatS, 0.0, A);
	return A;
}

gsl_matrix * inv_diagonalise(gsl_vector * v, int M, int N) {
	double vk;
	gsl_matrix * D = gsl_matrix_calloc(N, M);
	for (int k = 0; k < N; k++) {
		vk = 1 / (double) gsl_vector_get(v, k);
		gsl_matrix_set(D, k, k, vk);
	}
	return D;
}

gsl_matrix * diagonalise(gsl_vector * v, int M, int N) {
	double vk;
	gsl_matrix * D = gsl_matrix_calloc(M, N);
	for (int k = 0; k < N; k++) {
		vk = gsl_vector_get(v, k);
		gsl_matrix_set(D, k, k, vk);
	}
	return D;
}

gsl_vector * vector_truncate(gsl_vector * v, int M) {
	double vm;
	gsl_vector * vM = gsl_vector_alloc(M);
	for (int m = 0; m < M; m++){
		vm = gsl_vector_get(v, m);
		gsl_vector_set(vM, m, vm);
	}
	return vM;
}

gsl_matrix * matrix_extract(gsl_matrix * A, int M_start, int M_end, int N_start, int N_end) {
	int k = 0, l = 0;
	double Amn;
	gsl_matrix * AMN = gsl_matrix_calloc(M_end - M_start + 1, N_end - N_start + 1);
	for (int m = M_start; m < M_end + 1; m++) {
		for (int n = N_start; n < N_end + 1; n++) {
			Amn = gsl_matrix_get(A, m, n);
			gsl_matrix_set(AMN, k, l, Amn);
			l++;
		}
		l = 0, k++;
	}
	return AMN;
}

gsl_matrix * construct_SM_sqrt(gsl_vector * s, int M) {
	double sMi;
	gsl_vector * sM = gsl_vector_alloc(M);
	gsl_vector * sM_sqrt = gsl_vector_alloc(M);

	// Truncate the vector s to the Mth value
	sM = vector_truncate(s, M);

	// Take the square root of each entry and output to new vector
	for (int i = 0; i < M; i++) {
		sMi = gsl_vector_get(sM, i);
		gsl_vector_set(sM_sqrt, i, sqrt(sMi));
	}

	// Generate an M x M matrix from this vector
	return diagonalise(sM_sqrt, M, M);
}

gsl_matrix * construct_Hankel(double * hns, int N) {
	gsl_matrix * HN = gsl_matrix_alloc(N + 1, N + 1);
	for (int k = 0; k < N + 1; k++) {
		for (int l = 0; l < N + 1; l++)
			gsl_matrix_set(HN, k, l, hns[k + l]);
	}
	return HN;
}




	// for (int m = 0; m < M; m++) {
	// 	for (int n = 0; n < M; n++)
	// 		printf("m = %d, n = %d, SM_sqrt_mn = %.15lf\n", m, n, gsl_matrix_get(SM_sqrt, m, n));
	// }


	// gsl_matrix * SVT = gsl_matrix_alloc(N + 1, N + 1);
	// gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, S, V, 0.0, SVT);
	// gsl_matrix * HN_test = gsl_matrix_alloc(N + 1, N + 1);
	// gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, HN, SVT, 0.0, HN_test);



	// for (int k = 0; k < N + 1; k++) {
		// for (int l = 0; l < N + 1; l++)
			// printf("k = %d, l = %d, H_kl = %.15lf\n", k, l, gsl_matrix_get(HN, k, l));
	// }

	// double ** HN = (double **) malloc((N + 1) * sizeof(double *));
	// for (int k = 0; k < N + 1; k++)
	// 	HN[k] = (double *) malloc((N + 1) * sizeof(double));