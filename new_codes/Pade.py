import numpy as np
from scipy.special import gamma, factorial, dawsn, erf
from scipy import linalg as LA
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats

class Gen_Pade(object):

	def __init__(self, K, eps):
		self.K = K
		self.eps = eps
		self.eps_sq = eps ** 2

	def Kung(self, fks, gks):
		"""
		Computes alpha_k, gamma_k such that |h_k - sum_{j = 1}^L (alpha_j * gamma_j ^ k)| < eps for k = 1, ..., L, where L <= K. L is the dimension of the Hankel matrix H formed from h_k. To ensure sufficiently small singular values of H are found, K must be chosen to be sufficiently large.
		"""

		ext_len = 2 * self.K + 1
		if (len(fks) != ext_len) or (len(gks) != ext_len):
			raise ValueError("Length of coefficients arrays should equal 2K + 1\n")

		# Construct the ratio of the Maclaurin coefficients of f and g
		hks = np.empty(ext_len, dtype=np.complex128)
		for k in range(ext_len):
			hks[k] = fks[k] / gks[k]

		# Perform SVD on the Hankel matrix generated from the hks
		H = LA.hankel(hks[:self.K], hks[self.K:])
		U, Sigma, Vh = LA.svd(H)

		# Find L big enough for sufficient singular value decay of H.
		L = 1
		while np.sqrt(L) * Sigma[L + 1] >= self.eps_sq:
			L += 1

		# Form the realisation triple (A, b, c).
		U_tilde = U[:self.K - 1, :L]; U_hat = U[1:self.K, :L]; Sigma_L = np.diag(Sigma[:L])
		A = LA.pinv(U_tilde @ np.sqrt(Sigma_L)) @ U_hat @ np.sqrt(Sigma_L)
		b = np.sqrt(Sigma_L) @ Vh[:L, 0]
		c = U[0, :L] @ np.sqrt(Sigma_L)

		# Compute the alphas and gammas for j = 1, 2, ..., L from the eigendecomposition of A and the realisation triple (the gammas are just the eigenvalues of A).
		gammas, W = LA.eig(A)
		alphas = (c @ W).T * (LA.inv(W) @ b)
		return alphas, gammas

	def f_by_g(self, alphas, gammas, xL, xR, nx, g):

		L = len(alphas)
		xs = np.linspace(xL, xR, nx)
		G = np.empty((L, nx), dtype=np.complex128)
		for j in range(L):
			G[j] = alphas[j] * g(gammas[j] * xs)
		return np.sum(G, axis=0), xs


class Matrix_computation(object):

	def __init__(self, A, B, C, D, E, omega):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.E = E
		self.omega = omega

	def F(self, a, b, z):

		first_term = np.cos(a + b) * np.exp(-a ** 2 / (4.0 * z ** 2))

		second_term = erf(1j * a / (2.0 * z) + 2 * z) + erf(1j * a / (2.0 * z) - 2 * z) - 2 * erf(1j * a / (2.0 * z))

		third_term = 1j * np.sin(a + b) * np.exp(-a ** 2 / (4.0 * z ** 2))

		fourth_term = erf(1j * a / (2.0 * z) + 2 * z) - erf(1j * a / (2.0 * z) - 2 * z)

		return first_term * second_term + third_term * fourth_term


	def exact_entries(self, N):

		A_mat = np.empty((2 * N + 1, 2 * N + 1), dtype=np.complex128)
		# A_mat = np.empty((101, 101), dtype=np.complex128)

		ms = np.array(range(-N, N + 1))
		ns = np.array(range(-N, N + 1))
		# ms = np.linspace(-N, N + 1, 101)
		# ns = np.linspace(-N, N + 1, 101)

		# Fox-Li case
		if (self.D == 1 and self.E == 1 and self.B == -2):
			
			# Construct the parameters of the integral
			a = self.omega * self.A + np.pi * ms
			b = self.omega * self.C - np.pi * ns
			# a = ms; b = -ns
			z = 1j * np.sqrt(1j * self.omega)
			aa, bb = np.meshgrid(a, b)
			pos_inds = tuple([aa + bb != 0])
			zero_inds = tuple([aa + bb == 0])
			aa_pos, bb_pos = aa[pos_inds], bb[pos_inds]

			A_mat[pos_inds] = 0.5 * np.pi ** (0.5) / (2 * 1j * z * (aa_pos + bb_pos)) * (self.F(aa_pos, bb_pos, z) - self.F(bb_pos, aa_pos, z))
			# A_mat[zero_inds] = 0.5 / (1j * self.omega) * (1 - np.exp(4 * 1j * self.omega))
			first_term = np.exp(4 * 1j * self.omega) / np.sqrt(np.pi)
			second_term = 1 / np.sqrt(np.pi)
			third_term = 1j * np.sqrt(1j * self.omega) * erf(2 * 1j * np.sqrt(1j * self.omega))
			fourth_term = 1j * np.sqrt(1j * self.omega) * erf(-2 * 1j * np.sqrt(1j * self.omega))
			A_mat[zero_inds] = np.sqrt(np.pi) * 1j / (2.0 * self.omega) * (first_term - second_term + third_term - fourth_term)
			print(A_mat)

		return A_mat



