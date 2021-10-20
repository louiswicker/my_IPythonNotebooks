"""
Solve Advection Equation using upwind scheme

	dU / dt + C dU / dx = 0


* U(x,t)
* C: 0.5
* Initial condition: U(x, 0) = exp( - x^2 )
"""
import numpy as np
from optparse import OptionParser

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import animation

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)

global time_text
global U
global xn
global tn

def solver(scheme='LaxW', nx = 100, C = 0.5):
	
	def init():
	
		line.set_data([], [])
		time_text.set_text('')
		return line,
	
	def animate(n):
	
		time_text.set_text('time = %.1f' % tn[n])
		line.set_data(xn, U[:, n])
	
		return line,


	T = 2.0 # t
	X = 10.0 # x
	
	nlabel = np.arange(6) / 5.0
	
	ntimes = nx * nlabel / np.abs(C)
	ntimes = ntimes.astype('int64')
	# print(ntimes)
	
	nt = ntimes[-1]
	# print(nx)
	# print(nt)
	
	
	dt = T / ntimes[-1]   # dt
	dx = X / (nx - 1)     # dx
	# print(dt)
	# print(dx)
	
	U = np.zeros((nx, nt+1))
	
	# Set the initial values
	for i in np.arange(0, nx):
		U[i, 0] = np.exp(- ((X*0.5 - i * dx)) ** 2)
	
	# for j in np.arange(1, m):
	# 	for i in np.arange(1, n):
	# 		U[i, j] = (k * U[i, j - 1] + C * h * U[i - 1, j]) / (k + C * h)

	if scheme == 'LaxW':			
		for n in np.arange(1, nt+1):
			U[1:nx-1, n] = U[1:nx-1,n-1] - 0.5*C * (U[2:nx,n-1] - U[0:nx-2,n-1]) + 0.5*(C**2)*((U[2:nx,n-1] - 2*U[1:nx-1,n-1] + U[0:nx-2,n-1]))
				
			# Periodic BCs
			U[0,   n] = U[nx-2,n]
			U[nx-1,n] = U[1,   n]
	
	else:			
		for n in np.arange(1, nt+1):
			U[1:nx-1, n] = U[1:nx-1,n-1] - 0.5*C * (U[2:nx,n-1] - U[0:nx-2,n-1]) + 0.5*abs(C)*((U[2:nx,n-1] - 2*U[1:nx-1,n-1] + U[0:nx-2,n-1]))
	
			# Periodic BCs
			U[0,   n] = U[nx-2,n]
			U[nx-1,n] = U[1,   n]
	
	tn = np.zeros((nt+2, 1))
	for n in np.arange(0, nt+2):
		tn[n] = n * dt
	
	xn = np.zeros((nx, 1))
	
	for i in np.arange(0, nx):
		xn[i] = i * dx
	
	fig = plt.figure(1)
	
	for n in ntimes:
		subfig = fig.add_subplot(1, 1, 1)
		label = 't = ' + str(tn[n][0])
		subfig.plot(xn, U[:, n], label=label)
		subfig.legend()
		subfig.grid(True)
		print(n, label)
	
	
	# Save Image
	plt.xlabel('x: position')
	plt.ylabel('u: u(x, t)')
	plt.title(scheme)
	#plt.savefig('transport-equation')
	
	
	fig = plt.figure()
	ax = plt.axes(xlim=(0, X), ylim=(-1, 1.5))
	ax.grid()
	
	line, = ax.plot([], [], lw=2)
	
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=ntimes[-1], interval=10, blit=True)
	anim.save('transport-equation.mp4', fps=int(ntimes[-1]/20), extra_args=['-vcodec', 'libx264'])

# def main():
# 	#---------------------------------------------------------------------------------------------------
# 	# Main function defined to return correct sys.exit() calls
# 	#
# 	parser = OptionParser()
# 	parser.add_option("--scheme", dest="scheme", type="string", default="Upwind", \
# 								  help="Name of advection scheme")
# 	parser.add_option("--nx", dest="nx",   type="int",  default=100, help = "Number of horizontal points")
# 	parser.add_option("--cr", dest="cr",   type="float",  default=0.2, help = "Courant number")
# 	(options, args) = parser.parse_args()
	
# 	solver(scheme=options.scheme, nx = options.nx, C = options.cr)
	
# if __name__ == "__main__":
# 	main()




