#
# Course Project #2: Optimal Control of an Actuated Flexible Surface
# Group 17: Faedo Nicolò, Paltrinieri Mirco, Sabatini Luca
# Bologna, December 2024
#
# Main file

###############
#import section
###############
import numpy as np
import scipy as sp

from scipy.optimize import root

from dynamics import dynamics
from cost import cost_fcn

import armijo as arm 
import trajectory as trj 
import matplotlib.pyplot as plt

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

###########
#Parameters
###########


########
#Main
########

# # Even with a stopping critheria is good to have a maximum number of iterations
# max_iters = 5
# stepsize = 0.01
# n_x = 8           # state variable dimension

# cc =0.5     # Used only for armijo rule, not here


# # Initialize state, cost and gradient variables (for each algorithm)
# zz = np.zeros((n_x, max_iters))
# ll = np.zeros((max_iters-1))
# dl = np.zeros((n_x, max_iters-1))
# dl_norm = np.zeros(max_iters-1) #[for plots]

# # Set initial condition for each state variable
# zz_init = [-0.2,-0.01,0.05,-0.15,0,0,0,0]
# zz[:,0] = zz_init 

# # Algorithm
# for kk in range(max_iters-1):

#     # Compute cost and gradient
#     ll[kk], dl[:,kk] = cost_fcn(zz[:,kk]) 
#     dl_norm[kk] = np.linalg.norm(dl[:,kk]) #[for plots]

#     # select the direction
#     direction = - dl[:,kk]  # Direction is minus the gradient

#     ############################
#     # Descent plot
#     ############################

#     steps = np.linspace(0,1,int(1e3))
#     costs = np.zeros(len(steps))

#     for ii in range(len(steps)):

#         step = steps[ii]

#         zzp_temp = zz[:,kk] + step*direction   # temporary update

#         costs[ii] = cost_fcn(zzp_temp)[0]

#     plt.figure()

#     plt.clf()
#     plt.title('Descent')
#     plt.plot(steps, costs, color='g', label='$\\ell(z^k + stepsize*d^k$)')
#     plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps, color='r', label='$\\ell(z^k) + stepsize*\\nabla\\ell(z^k)^{\\top}d^k$')
#     plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(z^k) + stepsize*c*\\nabla\\ell(z^k)^{\\top}d^k$')
#     plt.grid()
#     plt.legend()

#     plt.show()

#     ############################
#     ############################

#     # Update the solution
#     zz[:,kk+1] = zz[:,kk] + stepsize * direction
#     print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,zz[:,kk+1]))

#     if np.linalg.norm(direction) <= 1e-4:
        
#         max_iters = kk+1

#         break


# plt.figure()
# plt.rcParams.update({'font.size': 12})
# domain_x = np.arange(-3,3,0.1)
# domain_y = np.arange(-3,3,0.1)
# domain_x, domain_y = np.meshgrid(domain_x, domain_y)
# cost_on_domain = np.zeros(domain_x.shape)

# for ii in range(domain_x.shape[0]):
#     for jj in range(domain_x.shape[1]):
#         cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_x[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

# ax = plt.axes(projection='3d')
# ax.plot_surface(domain_x, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.8)
# ax.plot3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange')
# ax.scatter3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange', s=50)
# ax.set_xlabel('$z^k_0$')
# ax.set_ylabel('$z^k_1$')
# ax.set_zlabel('$\ell(z^k)$')

# plt.show()

# Define the equilibrium function
def equilibrium_function(state_u):
    """
    Combines state and inputs into a single vector for optimization.
    """
    xx = state_u[:8]  # Extract state
    uu = state_u[8:]  # Extract inputs
    xx_next, _, _ = dynamics(xx, uu)  # Compute next state using dynamics
    return xx_next - xx  # Equilibrium condition: x_{t+1} - x_t = 0

# Initial guess for state and inputs
initial_state = np.zeros(8)  # 8 state variables (positions + velocities)
initial_input = np.zeros(2)  # 2 input variables (forces at points 2 and 4)
initial_guess = np.concatenate((initial_state, initial_input))

# Find equilibrium using root-finding
solution = root(equilibrium_function, initial_guess)

# Extract equilibrium state and input
equilibrium_state = solution.x[:8]
equilibrium_input = solution.x[8:]

# Print results
print("Equilibrium State:", equilibrium_state)
print("Equilibrium Input:", equilibrium_input)
print("Success:", solution.success)