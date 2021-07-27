import numpy as np
from scipy.special import gamma, factorial, dawsn
from scipy import linalg as LA
import matplotlib.pyplot as plt
import time


def hns_gen(N):

	"""
	Returns a vector of length 2N + 1 of the ratio of the Maclaurin coefficients of the
	analytic functions under observation.
	"""

	hns = np.zeros(2*N + 1)
	for n in np.arange(len(hns)):
		if n < 200:
			hns[n] = gamma(.5*(n + 2))/factorial(n + 1)
		# print(hns[n], n)
		# for k in np.arange(1, n + 2):
			# hns[n] /= (n + k)
	return hns


def Kung(N, eps_sq):

	"""
	This is an algorithm that returns alpha and gamma values that satisfy a relaxed moment
	problem up to a given error tolerance. The routine proceeds by forming the Hankel matrix
	that of the ratio of the Maclaurin coefficients of the target function f and the 
	approximant g.

	Given N, generates the Maclaurin coefficients of length 2N + 1 using hns_gen and
	then forms the Hankel matrix of dimension (N + 1) x (N + 1) from this. N must be
	chosen to be sufficiently large in order for M to be found that satisfies the
	tolerance constraint based on the singular values of H.
	"""

	#### Perform SVD on the Hankel matrix generated from the hns
	hns = hns_gen(N)
	H = LA.hankel(hns[:N + 1], hns[N:])
	# start = time.time()
	U, Sigma, Vh = LA.svd(H)
	# print("U = ")
	# print(U)
	# print("Sigma = ")
	# print(Sigma)
	# print("V = ")
	# print(Vh.T)
	# print('SVD computation time = ', time.time() - start)
	# print(Sigma)

	#### Find M big enough for sufficient singular value decay of H.
	M = 1
	while np.sqrt(M)*Sigma[M + 1] >= eps_sq:
		M += 1

	#### Use this M to form the realisation triple (A, b, c).
	# start = time.time()
	U_tilde = U[:N - 1, :M]; U_hat = U[1:N, :M]; Sigma_M = np.diag(Sigma[:M])
	A = LA.pinv(U_tilde@np.sqrt(Sigma_M))@(U_hat@np.sqrt(Sigma_M))
	b = np.sqrt(Sigma_M)@Vh[:M, 0]
	c = U[0, :M]@np.sqrt(Sigma_M)
	print(U_hat)
	# print("b values:")
	# print(b)
	# print("c values:")
	# print(c)
	# print('Inverse computation time = ', time.time() - start)

	#### Compute the alphas and gammas for m = 1, 2, ..., M from the eigendecomposition of 
	#### A and the realisation triple (the gammas are just the eigenvalues of A).
	gammas, W = LA.eig(A)
	alphas = (c@W).T*(LA.inv(W)@b)
	return alphas, gammas

# hns = hns_gen(N)
# for n in np.arange(2*N + 1):
# 	print(np.abs(hns[n] - np.dot(alphas, gammas**n)))

def approx_sinc(N, eps, xs):

	"""
	Appproximates sinc(x) as a sum of Gaussians, with parameters generated from the
	Kung algorithm.
	"""

	alphas, gammas = Kung(N, eps)

	#### The even case
	# gammas = np.sqrt(gammas) ####

	E = np.empty((len(alphas), len(xs)), dtype=np.complex128)
	for m in np.arange(len(alphas)):
		E[m] = np.exp(-(gammas[m]*xs)**2) + 1j*2/np.sqrt(np.pi)*dawsn(gammas[m]*xs)
	return np.sum(E*np.tile(alphas, (len(xs), 1)).T, axis=0)


def my_sinc(xs):
	sinc = np.empty(len(xs))
	for i in np.arange(len(xs)):
		if xs[i] == 0:
			sinc[i] = 1
		else:
			sinc[i] = np.sin(xs[i])/xs[i]
	return sinc


def approx_ind(N, eps_sq, xis):

	"""
	Approximates the indicator function over (-1, 1) as a sum of Gaussians. The
	parameters for these Gaussians come from the alphas and gammas that
	approximate sinc by a sum of Gaussians using Kung's algorithm.
	"""

	#### Form the parameters from those for the sinc approximation. These are
	#### available in closed form from the Fourier transform results for Gaussians.
	alphas, gammas = Kung(N, eps_sq)

	## Not even case (we square because the Fourier transform in this case is of 
	## e^(-gammas^2x^2))
	gammas = gammas**2

	alphas = alphas/(np.sqrt(np.pi*gammas))
	gammas = 1/(4*gammas)

	#### Form the Gaussian approximation of the indicator function.
	X = np.empty((len(alphas), len(xis)), dtype=np.complex128)
	for m in np.arange(len(alphas)):
		X[m] = np.exp(-gammas[m]*xis**2)
	return np.sum(X*np.tile(alphas, (len(xis), 1)).T, axis=0)


def Amn(N, eps_sq, omega):

	"""
	Generates the matrix entries that approximate the eigenvalues. The number of
	rows is 2M + 1 and the number of columns is 2N + 1, where -M <= m <= M and
	similar for n. Omega is the (Fresnel number?) of the Fox-Li.
	"""

	#### Generate parameters for Pade approximation of sinc by Gaussian.
	alphas, gammas = Kung(N, eps_sq)

	#### Transform these parameters according to the Fourier transform result.
	alphas = alphas/(np.sqrt(np.pi)*gammas)
	gammas = 1/(4*gammas**2)

	#### Matrix should be square for the eigendecomposition.
	M = N

	#### Construct intermediate variables for the matrix entries
	K = len(alphas); L = K
	az = 1j*omega - gammas
	cs = np.empty((K, 2*M + 1), dtype=np.complex128)
	ds = np.empty((K, 2*M + 1, 2*N + 1), dtype=np.complex128)
	for l in np.arange(L):
		cs[l] = az[l].real*omega*np.pi*np.arange(-N, N + 1)/(np.abs(az[l])**2)
		for m in np.arange(2*M + 1):
			ds[l, m] = (np.pi*(m - M)*np.abs(az[l])**2 - \
				omega*np.pi*np.arange(-N, N + 1)*az[l].imag)/(np.abs(az[l])**2)

	#### Compute the entries.
	A = np.zeros((2*M + 1, 2*N + 1), dtype=np.complex128)
	for m in np.arange(2*M + 1):
		for n in np.arange(2*N + 1):
			for k in np.arange(K):
				for l in np.arange(L):
					# A[m, n] += np.pi/2*alphas[k]*alphas[l]/(np.sqrt(omega**2 + az[l]*az[k]))\
					# *np.exp(np.pi**2*(n - N)**2/(4*az[l]))\
					# *np.exp((-(cs[l, n] + 1j*ds[l, m, n])**2)\
					# 	/(4*(omega**2/az[l] + az[k])))
					A[m, n] += np.pi/2*alphas[k]*alphas[l]/(np.sqrt(omega**2 + az[l]*az[k]))\
					*np.exp(np.pi**2*(m - M)**2/(4*az[k]))\
					*np.exp((-(cs[k, m] + 1j*ds[k, m, n])**2)\
						/(4*(omega**2/az[k] + az[l])))
	return A


N = 100
omega = 200
eps_sq = 1e-15
# xs = np.linspace(-10*np.pi, 10*np.pi, 5000)
# xis = np.linspace(-2*np.pi, 2*np.pi, 5000)
# y = approx_sinc(N, eps_sq, xs)
ax = plt.subplot(111)
# ax.plot(xs, y.real)
# ax.plot(xs, my_sinc(xs))
# chi_tilde = approx_ind(N, eps_sq, xis)
# plt.plot(xis, chi_tilde.real)
A = Amn(N, eps_sq, omega)
eigs = LA.eig(A)[0]
for lamb in np.sort(eigs):
	ax.plot(lamb.real, lamb.imag, '.')
ax.set_aspect('equal')
plt.show()



















