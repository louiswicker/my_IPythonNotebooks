#
import numpy as np
import matplotlib.pyplot as plt

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in np.arange(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in np.arange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

def para_spline(input, lbc, ubc, full_solution=True):

	nz = input.shape[0] - 1
	rhs = 3.0*(input[1:] + input[:-1])

	aa  = np.ones((nz,))
	bb  = 4.0*np.ones((nz,))
	cc  = np.ones((nz,))

	if lbc != None:
		rhs[0] = rhs[0] + aa[0]*lbc
	if ubc != None:
		rhs[-1] = rhs[-1] + cc[-1]*ubc

	if not full_solution:
		return TDMAsolver(aa, bb, cc, rhs)

	soln = TDMAsolver(aa, bb, cc, rhs)

	return np.concatenate([[lbc],soln,[ubc]])



if __name__ == "__main__":
	x = np.linspace(0.0, 1.0, num=10, endpoint=True)
	array = np.sin(x * 2.0*np.pi)
	matrix = para_spline(array, 10.0, -10.0)
	print(matrix, matrix.shape)
	plt.plot(array)
	plt.show()
	plt.plot(matrix)
	plt.show()
