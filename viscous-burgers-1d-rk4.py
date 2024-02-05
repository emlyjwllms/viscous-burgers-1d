# solve u_t + u u_x - nu u_xx = 0 with 2nd order forward FD in space and RK4 in time
# nu = 0.1

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 16
rcParams["figure.figsize"] = (8,4)
rcParams['figure.dpi']= 200
rcParams["figure.autolayout"] = True


# physical parameters
nu = 0.1 # diffusion

# spatial mesh
dx = 0.05
x0 = -1
xf = 3
x = np.arange(x0,xf+dx,dx)
nx = len(x)

# temporal mesh
dt = 0.01
t0 = 0
tf = 1
t = np.arange(t0,tf+dt,dt)
nt = len(t)

# Cole-Hopf
def phi(x,t):
    return 4 + np.exp(-nu*np.pi**2*t) * np.cos(np.pi*x)

def dphidx(x,t):
    return -np.pi*np.exp(-nu*np.pi**2*t)*np.sin(np.pi*x)

# initial condition
u0 = -2*nu*dphidx(x,0)/phi(x,0)

# solution structures
u_approx = np.empty((nx,nt))
u_analytical = np.empty((nx,nt))
u_approx[:,0] = u0
u_analytical[:,0] = u0

# plot initial condition
plt.plot(x,u_approx[:,0],'k.',alpha=0.1)
plt.plot(x,u_analytical[:,0],'k',alpha=0.1)

def f(u,t):
    A = (nu - 0.5*u*dx)*np.eye(nx-2,k=1) + (-2*nu)*np.eye(nx-2) + (nu + 0.5*u*dx)*np.eye(nx-2,k=-1)
    return np.matmul(A,u)/dx**2

# time integration
for n in range(0,nt-1):

    # RK4 time-stepping
    k1 = f(u_approx[1:-1,n], t[n])
    k2 = f(u_approx[1:-1,n] + k1*dt/2, t[n] + dt/2)
    k3 = f(u_approx[1:-1,n] + k2*dt/2, t[n] + dt/2)
    k4 = f(u_approx[1:-1,n] + k3*dt, t[n] + dt)
    u_approx[1:-1,n+1] = u_approx[1:-1,n] + (k1 + 2*k2 + 2*k3 + k4)*dt/6

    # dirichlet BCs
    u_approx[0,n+1] = 0
    u_approx[-1,n+1] = 0

    # analytical solution
    u_analytical[:,n+1] = -2*nu*dphidx(x,t[n])/phi(x,t[n])
    
    if n%10 == 0 and n < 50:
        plt.plot(x,u_approx[:,n],'k.',alpha=n/50)
        plt.plot(x,u_analytical[:,n],'k-',alpha=n/50)
    if n == 50:
        plt.plot(x,u_approx[:,n],'k.',alpha=n/50, label="numerical")
        plt.plot(x,u_analytical[:,n],'k-',alpha=n/50, label="analytical")

plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend(loc="upper left")
plt.title('1D viscous Burgers')
#plt.xlim(1,5)
plt.savefig("burgers-dirichlet-rk4.png",dpi=300,format='png',transparent=True)
plt.show()