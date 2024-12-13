#
# Course Project #2: Optimal Control of an Actuated Flexible Surface
# Group 17: Faedo Nicolò, Paltrinieri Mirco, Sabatini Luca
# Bologna, December 2024
#
# Dynamics file

import numpy as np
from scipy.spatial.distance import euclidean

ns = 8  # State dimension
ni = 2  # Input dimension

n_points = 4  # Number of points in the surface

dt = 1e-3 # discretization stepsize - Forward Euler

m = 0.1
m_act = 0.2
m_i = [m, m_act, m, m_act]

d = 0.2
alpha = 128*0.25
c = 0.1

def dynamics(xx,uu):
    """
    Dynamics of a discrete-time Actuated Flexible Surface

    Args
        - xx \in \R^8 state at time t
        - uu \in \R^2 input at time t

    Return 
        - next state xx_{t+1}
        - gradient of f wrt x, at xx,uu
        - gradient of f wrt u, at xx,uu

    Note
        - AA = dxf'
        - BB = duf' 
    """
    xx = xx[:,None]
    uu = uu[:,None]

    xxp = np.zeros((ns,1))
    dxf = np.zeros((ns, ns))
    duf = np.zeros((ns, ni))

    # Compute dynamics for each point
    for i in range(n_points):
        z_i = xx[i, 0]
        v_i = xx[i+n_points, 0]

        # Determine the force for the current point
        if i == 1:  # Point 2 (actuated)
            F_i = uu[0, 0]
        elif i == 3:  # Point 4 (actuated)
            F_i = uu[1, 0]
        else:  # Non-actuated points
            F_i = 0

        # Summation term for interactions with all points (i != j)
        summation = 0
        for j in range(n_points):
            if i != j:
                z_j = xx[j, 0]
                Lij = euclidean([abs(i - j)*d], [z_i - z_j])
                # Updated summation formula
                summation += (z_i - z_j) / (Lij * (Lij ** 2 - (z_i - z_j) ** 2))

        # Update position (z_i) and velocity (v_i) using Euler's method
        xxp[i] = z_i + dt * v_i
        xxp[i+n_points] = v_i + dt * (1 / m_i[i]) * (F_i - alpha * summation - c * v_i)

        # Gradients for position (z_i)
        dxf[i, i] = 1  # dz_i/dz_i
        dxf[i, i + n_points] = dt  # dz_i/dv_i

        # Gradients for velocity (v_i)
        dxf[i+n_points, i] = dt * (-alpha / m_i[i]) * (sum(
            (1 / (Lij * (Lij ** 2 - (z_i - z_j) ** 2)) for j in range(n_points) if j != i)))  # dv_i/dz_i
        dxf[i + n_points, i + n_points] = 1 - dt * c / m_i[i]  # dv_i/dv_i

        # Gradients for control inputs
        if i == 1:  # Point 2 (actuated)
            duf[i + n_points, 0] = dt / m_i[i]
        elif i == 3:  # Point 4 (actuated)
            duf[i + n_points, 1] = dt / m_i[i]

    xxp = xxp.squeeze()  # Convert back to vector format
    return xxp, dxf, duf
