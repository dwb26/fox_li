import numpy as np
from scipy.special import gamma, factorial, dawsn
from scipy import linalg as LA
import matplotlib.pyplot as plt
import cmath

def main():

	ifile = open("Amn.txt", "r")
	first_line = ifile.readline()
	first_line = first_line.split(",")
	N = int(first_line[0])
	omega = float(first_line[1])
	A_tilde = np.empty((2*N + 1)**2, dtype=np.complex128)
	n = 0
	for line in ifile:
		a, b = line.split(",")
		A_tilde[n] = complex(float(a), float(b))
		n += 1
	A_tilde = A_tilde.reshape((2*N + 1, 2*N + 1))
	eigs, vs = np.linalg.eig(A_tilde)
	eigs = np.sort(eigs)
	ax = plt.subplot(111)
	ax.plot(eigs.real, eigs.imag, 'b.')
	ax.set_title("N = {}, omega = {}".format(N, omega))
	plt.gca().set_aspect('equal')
	plt.savefig("N={}2.png".format(N))
	plt.show()

if __name__ == "__main__":
	main()
