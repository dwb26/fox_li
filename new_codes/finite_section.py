import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def F(a, b, z):

	first_term = np.cos(a + b) * np.exp(-a ** 2 / (4.0 * z ** 2))

	second_term = erf(1j * a / (2.0 * z) + 2 * z) + erf(1j * a / (2.0 * z) - 2 * z) - 2 * erf(1j * a / (2.0 * z))

	third_term = 1j * np.sin(a + b) * np.exp(-a ** 2 / (4.0 * z ** 2))

	fourth_term = erf(1j * a / (2.0 * z) + 2 * z) - erf(1j * a / (2.0 * z) - 2 * z)

	return first_term * second_term + third_term * fourth_term


def theta(a_vec, b_vec, z):

	L = len(a_vec)
	out = np.empty((L, L), dtype=np.complex128)
	a_count = 0
	b_count = 0
	for a in a_vec:
		b_count = 0
		for b in b_vec:
			if (a + b == 0):
				m1 = np.sqrt(np.pi) / z * np.exp(-a ** 2 / (4 * z ** 2))
				t1 = erf(1j * a / (2 * z) + 2 * z) - erf(1j * a / (2 * z) - 2 * z)
				t2 = (np.exp(-4 * z ** 2) * np.cos(2 * a) - 1) / z ** 2
				m2 = np.sqrt(np.pi) * 1j * a / (4 * z ** 3) * np.exp(-a ** 2 / (4 * z ** 2))
				t3 = erf(1j * a / (2 * z) + 2 * z) + erf(1j * a / (2 * z) - 2 * z) - 2 * erf(1j * a / (2 * z))
				out[a_count, b_count] = m1 * t1 + t2 + m2 * t3
			else:
				out[a_count, b_count] = np.sqrt(np.pi) / (2 * 1j * z * (a + b)) * (F(a, b, z) + F(b, a, z))
			b_count += 1
		a_count += 1
	return out


def theta_scalar(a, b, z):

	if a + b == 0:

		m1 = np.sqrt(np.pi) / z * np.exp(-a ** 2 / (4 * z ** 2))
		t1 = erf(1j * a / (2 * z) + 2 * z) - erf(1j * a / (2 * z) - 2 * z)
		t2 = (np.exp(-4 * z ** 2) * np.cos(2 * a) - 1) / z ** 2
		m2 = np.sqrt(np.pi) * 1j * a / (4 * z ** 3) * np.exp(-a ** 2 / (4 * z ** 2))
		t3 = erf(1j * a / (2 * z) + 2 * z) + erf(1j * a / (2 * z) - 2 * z) - 2 * erf(1j * a / (2 * z))
		return m1 * t1 + t2 + m2 * t3

	return np.sqrt(np.pi) / (2 * 1j * z * (a + b)) * (F(a, b, z) + F(b, a, z))


def construct_matrix(N, omega):

	A0 = np.zeros((2 * N, 2 * N), dtype=np.complex128)
	A1 = np.zeros((2 * N, 2 * N), dtype=np.complex128)
	z = np.sqrt(-1j * omega)

	short_indices = np.array(range(0, N))
	indices = np.array(range(0, 2 * N))
	even_indices = np.array(range(0, 2 * N, 2))
	odd_indices = np.array(range(1, 2 * N, 2))
	a = np.pi * short_indices
	c = np.pi * (short_indices - 0.5)
	aa, bb = np.meshgrid(a, a)
	cc, dd = np.meshgrid(c, c)


	# --------------------------------------------------------------------------------------------------------------- #
	#
	# Non-diagonal entries
	#
	# --------------------------------------------------------------------------------------------------------------- #

	A0[even_indices, 0::2] = (F(aa, bb, z) + F(bb, aa, z)) / (4 * 1j * np.sqrt(np.pi) * z * (aa + bb) / np.pi) + (F(aa, -bb, z) + F(-bb, aa, z)) / (4 * 1j * np.sqrt(np.pi) * z * (aa - bb) / np.pi)
	A0[even_indices, even_indices] = 0
	A0[odd_indices, 1::2] = -0.25 * (theta(c, c, z) - theta(-c, c, z) - theta(c, -c, z) + theta(-c, -c, z))
	A0[odd_indices, odd_indices] = 0


	# --------------------------------------------------------------------------------------------------------------- #
	#
	# Diagonal entries
	#
	# --------------------------------------------------------------------------------------------------------------- #

	for i in range(N):
		A1[2 * i, 2 * i] = 0.5 * (theta_scalar(a[i], a[i], z) + theta_scalar(a[i], -a[i], z))
		A1[2 * i + 1, 2 * i + 1] = 0.5 * (theta_scalar(c[i], -c[i], z) - theta_scalar(c[i], c[i], z))
	one = 1 / z * (2 * z * erf(2 * z) + 1 / np.sqrt(np.pi) * np.exp(-4 * z ** 2) - 1 / np.sqrt(np.pi))
	two = 1 / z * (2 * z * erf(-2 * z) - 1 / np.sqrt(np.pi) * np.exp(-4 * z ** 2) + 1 / np.sqrt(np.pi))
	integral = np.sqrt(np.pi) / (2 * z) * (one - two)
	A1[0, 0] = np.sqrt(np.pi) / z * (erf(2 * z) - erf(-2 * z))
	# A1[0, 0] = np.sqrt(np.pi) / (2 * z) * (one - two)

	return A0 + A1


N = 250
omega = 100
A = construct_matrix(N, omega)
eigs, _ = np.linalg.eig(A)
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.plot(eigs.real, eigs.imag, ".")
plt.show()






