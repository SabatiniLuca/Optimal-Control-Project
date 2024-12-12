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

#discretization step
dstep = 1e-3

#distance between 
d=0.2

def dynamics(xx,uu):

    # xx = [: , None]
    # uu = [: , None]

    m = [0.1 , 0.2 , 0.1 , 0.2]
    ##TODO alpha
    xxs = np.zeros((nstate,1))

    xxs[0] = xx[0] + dstep * xx[4]
    xxs[4] = xx[4] + dstep * 1/m[0] * (- alpha * sum(xx,1) - c * xx[4])

    xxs[1] = xx[1] + dstep * xx[5]
    xxs[5] = xx[5] + dstep * 1/m[1] * (- alpha * sum(xx,2) - c * xx[5])

    xxs[2] = xx[2] + dstep * xx[6]
    xxs[6] = xx[6] + dstep * 1/m[2] * (- alpha * sum(xx,3) - c * xx[6])

    xxs[3] = xx[3] + dstep * xx[7]
    xxs[7] = xx[7] + dstep * 1/m[3] * (- alpha * sum(xx,4) - c * xx[7])
        
    ##########
    # Gradient
    ##########

    #initialization

    dfx = np.zeros((ns, ns))
    dfu = np.zeros((ni, ns))

        
    #df1
    dfx[0,0] = 1
    dfx[1,0] = dt

    dfu[0,0] = 0

    #df2

    dfx[0,1] = dt*-gg / ll * np.cos(xx[0,0])
    dfx[1,1] = 1 + dt*(- kk / (mm * ll))

    dfu[0,1] = dt / (mm * (ll ** 2))

    xxp = xxp.squeeze()

    return xxp, dfx, dfu





def sum(xx,iter):
    temp=0
    for i in range(4) and i != iter:
        temp += (xx[iter]-xx[i])/(Lij(iter,i)*(Lij(iter,i)**2 - (xx[iter] - xx[i])**2))
    return temp


def Lij(iter,i):
    return np.sqrt((xx[iter]-xx[i])**2 + ((d*(iter-i))**2))
