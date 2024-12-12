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

from scipy.optimize import root

from dynamics import dynamics
from cost import cost_fcn
import armijo as arm 
import trajectory as trj 
import matplotlib.pyplot as plt
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

# Define the equilibrium function
def equilibrium_function(state):
    """
    Combines state and inputs into a single vector for optimization.
    """
    xx = state[:8]  # Extract state
    xx_next, _, _ = dynamics(xx, initial_input)  # Compute next state using dynamics
    return xx_next - xx  # Equilibrium condition: x_{t+1} - x_t = 0

# # Initial guess for state and inputs
initial_state = np.zeros(8)  # 8 state variables (positions + velocities)
# initial_input = np.zeros(2)  # 2 input variables (forces at points 2 and 4)
initial_input = np.array([1, -1])  # 2 input variables (forces at points 2 and 4)
# # initial_guess = np.concatenate((initial_state))

# # Find equilibrium using root-finding
# solution = root(equilibrium_function, initial_state)

# # Extract equilibrium state and input
# equilibrium_state = solution.x[:8]
# equilibrium_input = solution.x[2:]

# # Print results
# print("Equilibrium State:", equilibrium_state)
# print("Equilibrium Input:", equilibrium_input)
# print("Success:", solution.success)

# # Plot equilibrium state and input
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(equilibrium_state[:4], 'bo-', label='Positions')
# plt.plot(equilibrium_state[4:], 'ro-', label='Velocities')
# plt.title('Equilibrium State')
# plt.xlabel('State Index')
# plt.ylabel('Value')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(equilibrium_input, 'go-', label='Inputs')
# plt.title('Equilibrium Input')
# plt.xlabel('Input Index')
# plt.ylabel('Value')
# plt.legend()

# plt.tight_layout()
# plt.show()


# Find equilibrium using root-finding
solution = root(equilibrium_function, initial_state)

# Extract equilibrium state
equilibrium_state = solution.x

print("Equilibrium state:", equilibrium_state)

# Optionally, you can simulate the system from the initial state to see its evolution
time_horizon = 10  # seconds
dt = 1e-3  # time step
num_steps = int(time_horizon / dt)
states = np.zeros((num_steps, 8))
states[0, :] = initial_state

for t in range(1, num_steps):
    states[t, :], _, _ = dynamics(states[t-1, :], np.array([0.1, -0.1]))  # Natural evolution with no input

# Plot the state evolution
plt.figure()
for i in range(8):
    plt.plot(np.arange(num_steps) * dt, states[:, i], label=f'State {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.legend()
plt.grid()
plt.show()