#
# Course Project #2: Optimal Control of an Actuated Flexible Surface
# Group 17: Faedo Nicolò, Paltrinieri Mirco, Sabatini Luca
# Bologna, December 2024
#
# Dynamics file

import numpy as np

"""
Discretization of the system, using Forward Euler method
xt = xt + dt * xt'
"""

nstate = 8
ninput = 2

dstep = 1e-3

def dynamics(xx,uu):

    # xx = [: , None]
    # uu = [: , None]


    ##TODO masses vector
    ##TODO alpha
    xxs = np.zeros((nstate,1))

    xxs[0] = xx[0] + dstep * xx[4]
    xxs[4] = xx[4] + dstep * 1/m[0]*(- alpha * sum(xx,1) - c * xx[4])

    xxs[1] = xx[1] + dstep * xx[5]
    xxs[5] = xx[5] + dstep * 1/m[1]*(- alpha * sum(xx,2) - c * xx[5])

    xxs[2] = xx[2] + dstep * xx[6]
    xxs[6] = xx[6] + dstep * 1/m[2]*(- alpha * sum(xx,3) - c * xx[6])

    xxs[3] = xx[3] + dstep * xx[7]
    xxs[7] = xx[7] + dstep * 1/m[3]*(- alpha * sum(xx,4) - c * xx[7])

def sum(xx,iter):
    if (iter==1):
        temp=0
        for i in range(4) and i != iter:
            temp += (xx[iter]-xx[i])/(Lij(iter,i)*(Lij(iter,i)**2 - (xx[iter] - xx[i])**2))


def Lij(iter,i):
    return np.sqrt(())