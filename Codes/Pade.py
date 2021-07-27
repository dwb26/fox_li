import numpy as np
from scipy.special import gamma, factorial, dawsn
from scipy import linalg as LA
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats

def main():

	K = 30
	eps_sq = 1e-15

	# Generate the ratio of the Maclaurin coefficients #
	# ------------------------------------------------ #
	hns = hns_gen(K)

	# Solve the relaxed moment problem for sinc and Gaussian #
	# ------------------------------------------------------ #
	alphas, gammas = Kung(K, eps_sq)

	# Approximate the sinc function by Gaussian #
	# ----------------------------------------- #
	# approx_sinc(alphas, gammas)

	# Approximate the indicator function by Gaussian #
	# ---------------------------------------------- #
	approx_ind(alphas, gammas)

	# Output the alphas and gammas for the sinc approximation to a text file for the C operations #
	# ------------------------------------------------------------------------------------------- #
	output(alphas, gammas)

def output(alphas, gammas):
	"""
	Write the alphas and gammas to a text file for the C operations.
	"""
	alph_file = open("alphas.txt", "w")
	gamm_file = open("gammas.txt", "w")
	M = len(alphas)
	alph_file.write("{}\n".format(M))
	for m in np.arange(M):
		alph_file.write("{}, {}\n".format(alphas[m].real, alphas[m].imag))
		gamm_file.write("{}, {}\n".format(gammas[m].real, gammas[m].imag))
	alph_file.close()
	gamm_file.close()

def approx_ind(alphas, gammas):
	"""
	Approximates the indicator function over (-1, 1) as a sum of Gaussians. The parameters for these Gaussians come from the alphas and gammas that	approximate sinc by a sum of Gaussians by solving the relaxed moment problem.
	"""
	# Not even case (we square because the Fourier transform in this case is of e^(-gammas^2x^2))
	gammas = gammas ** 2

	alphas = alphas / (np.sqrt(np.pi * gammas))
	gammas = 1 / (4 * gammas)

	# Form the Gaussian approximation of the indicator function.
	xs = np.linspace(-3, 3, 5000)
	ys = np.linspace(-3, 3, 5000)
	def construct_ind(xs):
		X = np.empty((len(alphas), len(xs)), dtype=np.complex128)
		for m in np.arange(len(alphas)):
			X[m] = np.exp(-gammas[m] * xs **2)
		ind = np.sum(X * np.tile(alphas, (len(xs), 1)).T, axis=0)
		return ind

	def construct_true_ind(xs):
		true_ind_x = (-1 <= xs) & (xs <= 1)
		return true_ind_x

	ind = construct_ind(xs)
	true_ind = construct_true_ind(xs)
	
	gamma_hat = np.min(gammas.real)
	l1_alpha = LA.norm(alphas, ord=1)
	# print(gamma_hat, l1_alpha, 8 * l1_alpha / gamma_hat * np.sqrt(np.pi / 2) + 2)
	print(16 / 25 * (2 * l1_alpha / gamma_hat * np.sqrt(np.pi / 2) + 0.5))
	other_bound_a = (5 * np.sqrt(np.pi / 2) / (2 * gamma_hat) + 0.5) * l1_alpha + 0.5
	other_bound_b = (np.sqrt(np.pi / 2) / gamma_hat + 1) * l1_alpha + 1
	other_bound = other_bound_a * other_bound_b
	print(other_bound)

	fig = plt.figure(figsize=(18, 6))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	ax1.plot(xs, true_ind, label=r"$\chi_{(-1, 1)}(x)$")
	ax1.plot(xs, ind.real, label="approx.")
	ax2.plot(xs, np.abs(true_ind - ind.real), label=r"$|r_L(x)|$")
	ax2.plot(xs, 0.4 * stats.norm.pdf(xs, loc=-1, scale=0.25), label=r"$0.4 * \mathcal{N}(-1, (0.25)^2)$")
	ax2.plot(xs, 0.4 * stats.norm.pdf(xs, loc=1, scale=0.25), label=r"$0.4 * \mathcal{N}(1, (0.25)^2)$")
	ax1.legend(prop={"size": 14})
	ax2.legend(loc="upper right", prop={"size": 14})
	ax1.set_xlabel(r"$x$", Fontsize=18)
	ax2.set_xlabel(r"$x$", fontsize=18)
	ax1.tick_params(axis="x", labelsize=11)
	ax2.tick_params(axis="x", labelsize=11)
	ax1.tick_params(axis="y", labelsize=11)
	ax2.tick_params(axis="y", labelsize=11)
	plt.tight_layout()
	plt.show()

def hns_gen(N):
	"""
	Returns a vector of length 2N + 1 of the ratio of the Maclaurin coefficients of sinc and Gaussian.
	"""
	hns = np.zeros(2 * N + 1)
	for n in np.arange(len(hns)):
		if n < 200:
			hns[n] = gamma(0.5 * (n + 2)) / factorial(n + 1)
	return hns

def Kung(N, eps_sq):
	"""
	Given N, generates the Maclaurin coefficients of length 2N + 1 using hns_gen and forms the Hankel matrix HN of dimension (N + 1) x (N + 1). N must be chosen to be sufficiently large in order for M to be found that satisfies the tolerance constraint based on the singular values of H. Returns alpha and gamma values that satisfy a relaxed moment problem up to a given error tolerance.
	"""

	# Perform SVD on the Hankel matrix generated from the hns
	hns = hns_gen(N)
	H = LA.hankel(hns[:N + 1], hns[N:])
	U, Sigma, Vh = LA.svd(H)

	# Find M big enough for sufficient singular value decay of H.
	M = 1
	while np.sqrt(M) * Sigma[M + 1] >= eps_sq:
		M += 1

	# Form the realisation triple (A, b, c).
	U_tilde = U[:N - 1, :M]; U_hat = U[1:N, :M]; Sigma_M = np.diag(Sigma[:M])
	A = LA.pinv(U_tilde @ np.sqrt(Sigma_M)) @ U_hat @ np.sqrt(Sigma_M)
	b = np.sqrt(Sigma_M) @ Vh[:M, 0]
	c = U[0, :M] @ np.sqrt(Sigma_M)

	# Compute the alphas and gammas for m = 1, 2, ..., M from the eigendecomposition of A and the realisation triple (the gammas are just the eigenvalues of A).
	gammas, W = LA.eig(A)
	alphas = (c @ W).T * (LA.inv(W) @ b)
	return alphas, gammas

def approx_sinc(alphas, gammas):
	"""
	Approximates sinc(x) as a sum of Gaussians.
	"""
	xs = np.linspace(-10*np.pi, 10*np.pi, 5000)
	E = np.empty((len(alphas), len(xs)), dtype=np.complex128)
	for m in np.arange(len(alphas)):
		E[m] = np.exp(-(gammas[m]*xs)**2) + 1j*2 / np.sqrt(np.pi) * dawsn(gammas[m]*xs)
	sinc = np.sum(E*np.tile(alphas, (len(xs), 1)).T, axis=0)

	def my_sinc(xs):
		sinc = np.empty(len(xs))
		for i in np.arange(len(xs)):
			if xs[i] == 0:
				sinc[i] = 1
			else:
				sinc[i] = np.sin(xs[i])/xs[i]
		return sinc
	sinc_exact = my_sinc(xs)

	ax = plt.subplot(111)
	ax.plot(xs, sinc.real)
	ax.plot(xs, sinc_exact)
	plt.show()


if __name__ == "__main__":
	main()




