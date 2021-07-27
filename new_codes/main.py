import numpy as np
import matplotlib.pyplot as plt
from Pade import Gen_Pade, Matrix_computation
from scipy.special import gamma, factorial, dawsn


# ---------------------------------------------------------------------------------------------------------------------
#
# Parameters
#
# ---------------------------------------------------------------------------------------------------------------------
K = 60
eps = 1e-07
scaler = 1
xL = -scaler * np.pi; xR = scaler * np.pi
nx = 3000
A = 0
B = -2
C = 0
D = 1
E = 1
omega = 25
N = 500


# ---------------------------------------------------------------------------------------------------------------------
#
# Functions
#
# ---------------------------------------------------------------------------------------------------------------------

def gauss(xs):
	return np.exp(-xs ** 2)

def construct_fks(K):
	fks = np.empty(2 * K + 1, dtype=np.complex128)
	for k in range(2 * K + 1):
		fks[k] = (1j * np.pi) ** k / factorial(k + 1)
	return fks

def construct_gks(K):
	gks = np.empty(2 * K + 1, dtype=np.complex128)
	for k in range(2 * K + 1):
		gks[k] = 1j ** k / gamma(0.5 * (k + 2))
	return gks

def Fourier_transform_eqs(alphas, gammas):
	new_alphas = alphas * np.sqrt(np.pi / (gammas ** 2))
	new_gammas = np.pi ** 2 / (4.0 * gammas ** 2)
	return new_alphas, new_gammas


# ---------------------------------------------------------------------------------------------------------------------
#
# Indicator approximation
#
# ---------------------------------------------------------------------------------------------------------------------
# ind_app = True
ind_app = False

if ind_app:
	fks = construct_fks(K)
	gks = construct_gks(K)
	gp = Gen_Pade(K, eps)
	alphas, gammas = gp.Kung(fks, gks)
	sinc_approx, xs = gp.f_by_g(alphas, gammas, xL, xR, nx, gauss)
	ind = (np.abs(xs) < 1)
	alphas_t, gammas_t = Fourier_transform_eqs(alphas, gammas)
	ind_approx, _ = gp.f_by_g(alphas_t, np.sqrt(gammas_t), xL, xR, nx, gauss)


# ---------------------------------------------------------------------------------------------------------------------
#
# Exact entry computation
#
# ---------------------------------------------------------------------------------------------------------------------
mc = Matrix_computation(A, B, C, D, E, omega)
A_mat = mc.exact_entries(N)
eigs, _ = np.linalg.eig(A_mat)
# eigs = np.sort(eigs)[::-1]
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
ax.plot(eigs.real, eigs.imag, ".")
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
#
# Plotting
#
# ---------------------------------------------------------------------------------------------------------------------
# plot = True
plot = False

if plot:
	fig = plt.figure(figsize=(12, 8))
	ax1 = plt.subplot(411)
	ax2 = plt.subplot(412)
	ax3 = plt.subplot(413)
	ax4 = plt.subplot(414)
	ax1.plot(xs, np.sinc(xs), label="sinc(x)")
	ax1.plot(xs, sinc_approx.real, label="approximant")
	ax2.plot(xs, ind, label=r"$\chi_{(-1, 1)}(x)$")
	ax2.plot(xs, ind_approx.real, label="approximant")
	ax3.plot(xs, np.log10(np.abs(np.sinc(xs) - sinc_approx.real)), label=r"$\log_{10}$(|sinc_err|)")
	ax4.plot(xs, np.log10(np.abs(ind - ind_approx.real)), label=r"$\log_{10}$(|ind_err|)")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	plt.show()