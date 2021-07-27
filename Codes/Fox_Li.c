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

// void construct_A_tilde(_Complex double ** A_tilde, double omega, int N, _Complex double * alphas, _Complex double * gamma, int M);
void construct_A_tilde(_Complex double ** A_tilde, double omega, int s, _Complex double * alphas, _Complex double * gammas, int M, double * xi);

int main(void) {

	/* Read in alpha and gamma data */
	/* ---------------------------- */
	FILE * alpha_data;
	FILE * gamma_data;
	FILE * Amn;
	alpha_data = fopen("alphas.txt", "r");
	gamma_data = fopen("gammas.txt", "r");
	Amn = fopen("Amn.txt", "w");
	int M;
	double a1, b1, c1, d1;
	fscanf(alpha_data, "%d\n", &M);
	_Complex double * alphas = (_Complex double *) malloc(M * sizeof(_Complex double));
	_Complex double * gammas = (_Complex double *) malloc(M * sizeof(_Complex double));
	for (int m = 0; m < M; m++) {
		fscanf(alpha_data, "%lf, %lf\n", &a1, &b1);
		fscanf(gamma_data, "%lf, %lf\n", &c1, &d1);
		alphas[m] = CMPLX(a1, b1);
		gammas[m] = CMPLX(c1, d1);
	}

	/* Construct approximate matrix */
	/* ---------------------------- */
	int s;
	double b, h;
	// double omega = 2500.0;
	double omega = 50.0;
	if (omega * 1.1 < 1000)
		b = omega * 1.1;
	else
		b = 1000;
	if (M_PI * omega * 2.2 < 1000 * 2)
		s = (int) M_PI * omega * 2.2;
	else
		s = 1000 * 2;

	double * xi = (double *) malloc(s * sizeof(double));
	xi[0] = 0.0;
	h = (b - 0.0) / (s - 1);
	for (int i = 1; i < s; i++)
		xi[i] = xi[i - 1] + h;

	int N = 150;
	
	// _Complex double ** A_tilde = (_Complex double **) malloc((2*N + 1) * sizeof(_Complex double *));
	_Complex double ** A_tilde = (_Complex double **) malloc(s * sizeof(_Complex double *));
	// for (int n = 0; n < 2*N + 1; n++)
		// A_tilde[n] = (_Complex double *) malloc((2*N + 1) * sizeof(_Complex double));
	for (int n = 0; n < s; n++)
		A_tilde[n] = (_Complex double *) malloc(s * sizeof(_Complex double));

	construct_A_tilde(A_tilde, omega, s, alphas, gammas, M, xi);
	// construct_A_tilde(A_tilde, omega, N, alphas, gammas, M, xi);

	/* Write output */
	/* ------------ */
	fprintf(Amn, "%d, %lf\n", s, omega);
	// fprintf(Amn, "%d, %lf\n", N, omega);
	for (int m = 0; m < s; m++) {
		for (int n = 0; n < s; n++)
			fprintf(Amn, "%.15lf, %.15lf\n", creal(A_tilde[m][n]), cimag(A_tilde[m][n]));
	}
	// for (int m = 0; m < N; m++) {
		// for (int n = 0; n < N; n++)
			// fprintf(Amn, "%.15lf, %.15lf\n", creal(A_tilde[m][n]), cimag(A_tilde[m][n]));
	// }

	fclose(alpha_data);
	fclose(gamma_data);
	fclose(Amn);
	return 0;
}

void construct_A_tilde(_Complex double ** A_tilde, double omega, int s, _Complex double * alphas, _Complex double * gammas, int M, double * xi) {

	_Complex double iomega = _Complex_I * omega;
	_Complex double exp_num, exp_den, exp_term, exp_num1, exp_den1, exp_term1, coeff, Amn = 0 + 0*_Complex_I;
	_Complex double * f_alphas = (_Complex double *) malloc(M * sizeof(_Complex double));
	_Complex double * f_gammas = (_Complex double *) malloc(M * sizeof(_Complex double));
	_Complex double * as = (_Complex double *) malloc(M * sizeof(_Complex double));

	/* Apply the effects of the Fourier transform on the parameters */
	for (int m = 0; m < M; m++) {
		f_alphas[m] = alphas[m] / (_Complex double) (sqrt(M_PI) * gammas[m]);
		f_gammas[m] = 1.0 / (_Complex double) (4 * cpow(gammas[m], 2));
		as[m] = -f_gammas[m] + iomega;
	}	

	/* Compute the matrix terms */
	clock_t timer_start = clock();
	printf("Running the matrix calculations...\n");
	for (int m = 0; m < s; m++) {
		// printf("%02i*******************", m);
		for (int n = 0; n < s; n++) {
			for (int k = 0; k < M; k++) {
				for (int l = 0; l < M; l++) {
					coeff = (f_alphas[k]*f_alphas[l]) / (_Complex double) (csqrt(as[k]*as[l] + pow(omega, 2)));
					// exp_num = (as[k]*as[l] + pow(omega, 2))*pow(n, 2) - cpow(omega*n + 1j*m*as[l], 2);
					exp_num = (as[k]*as[l] + pow(omega, 2))*pow(xi[n], 2) - cpow(omega*xi[n] + 1j*xi[m]*as[l], 2);
					exp_den = 4*as[l]*(as[k]*as[l] + pow(omega, 2));
					exp_term = cexp(pow(M_PI, 2) * exp_num / exp_den);
					Amn += coeff * exp_term;
				}
			}
			A_tilde[m][n] = (Amn * M_PI) / 2.0;
			Amn = 0 + 0*_Complex_I;
		}
	}
	printf("Done in %f seconds.\n", (double) (clock() - timer_start) / (double) CLOCKS_PER_SEC);
}

























	/* Compute the matrix terms */
	// clock_t timer_start = clock();
	// printf("Running the matrix calculations...\n");
	// for (int m = -s; m < s + 1; m++) {
	// 	// printf("%02i*******************", m);
	// 	for (int n = -s; n < s + 1; n++) {
	// 		for (int k = 0; k < M; k++) {
	// 			for (int l = 0; l < M; l++) {
	// 				coeff = (f_alphas[k]*f_alphas[l]) / (_Complex double) (csqrt(as[k]*as[l] - pow(omega, 2)));
	// 				exp_num = pow(M_PI, 2) * ((as[k]*as[l] - pow(omega, 2))*pow(n, 2) + (omega * n - 1j*m*as[l]));
	// 				exp_den = 4*as[l] * (pow(omega, 2) - as[k]*as[l]);
	// 				exp_term = cexp(-exp_num / exp_den);
	// 				Amn += coeff * exp_term;
	// 			}
	// 		}
	// 		A_tilde[m + s][n + s] = (Amn * M_PI) / 2.0;
	// 		Amn = 0 + 0*_Complex_I;
	// 	}
	// }
	// printf("Done in %f seconds.\n", (double) (clock() - timer_start) / (double) CLOCKS_PER_SEC);


