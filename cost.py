#
# Course Project #2: Optimal Control of an Actuated Flexible Surface
# Group 17: Faedo Nicolò, Paltrinieri Mirco, Sabatini Luca
# Bologna, December 2024
#
# Costs file

# import dynamics as dyn

import numpy as np


def cost_fcn(xx):
    """
    Input:
    - state: Optimization variable (8 state variables) [z1, z1_dot, ..., xx[3], xx[3]_dot].
    
    Output:
    - cost: Scalar value of the cost function.
    - grad: Gradient of the cost function w.r.t. state variables.
    """
    
    # Cost components
    # Penalize deviations of non-actuated points from zero
    cost_non_actuated = xx[0]**2 + xx[2]**2  # Non-actuated positions
    cost_actuated = xx[1]**2 + xx[3]**2      # Actuated positions
    
    # Penalize differences between neighboring points (coupling term)
    coupling_term = ((xx[1] - xx[0])**2 + (xx[2] - xx[1])**2 + (xx[3] - xx[2])**2)
    
    # Penalize velocities (smooth dynamics)
    velocity_term = xx[4]**2 + xx[5]**2 + xx[6]**2 + xx[7]**2
    
    # Total cost
    cost = cost_non_actuated + cost_actuated + coupling_term + velocity_term
    
    # Gradients w.r.t. each state variable
    grad = np.zeros(8)
    
    # Gradients for positions
    grad[0] = 2 * xx[0] - 2 * (xx[1] - xx[0])       # z1
    grad[2] = 2 * xx[1] + 2 * (xx[1] - xx[0]) - 2 * (xx[2] - xx[1])  # xx[1]
    grad[4] = 2 * xx[2] + 2 * (xx[2] - xx[1]) - 2 * (xx[3] - xx[2])  # xx[2]
    grad[6] = 2 * xx[3] + 2 * (xx[3] - xx[2])       # xx[3]
    
    # Gradients for velocities
    grad[1] = 2 * xx[4]  # z1_dot
    grad[3] = 2 * xx[5]  # xx[1]_dot
    grad[5] = 2 * xx[6]  # xx[2]_dot
    grad[7] = 2 * xx[7]  # xx[3]_dot
    
    return cost, grad
