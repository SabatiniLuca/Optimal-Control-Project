import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import signal
from dynamics import dynamics  # Ensure you import your dynamics function

# Allow Ctrl-C to work despite plotting
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({'font.size': 22})

# Define the equilibrium function
def equilibrium_function(state):
    """
    Computes the difference between the next state and the current state.
    """
    xx = state  # Extract state
    uu = np.zeros(2)  # Assume no input (natural evolution)
    xx_next, _, _ = dynamics(xx, uu)  # Compute next state using dynamics
    return xx_next - xx  # Equilibrium condition: x_{t+1} - x_t = 0

# Initial guess for state
initial_state = np.zeros(8)  # 8 state variables (positions + velocities)

# Find equilibrium using root-finding
solution = root(equilibrium_function, initial_state)

# Extract equilibrium state
equilibrium_state = solution.x

print("Equilibrium state:", equilibrium_state)

# Optionally, you can simulate the system from the initial state to see its evolution
time_horizon = 1  # seconds
dt = 1e-3  # time step
num_steps = int(time_horizon / dt)
states = np.zeros((num_steps, 8))
states[0, :] = initial_state

# Apply initial input for the first few steps
initial_input_duration = 5  # Number of steps to apply the initial input
initial_input = np.array([0.01, -0.01])  # Initial input

# Maximum length constraint
max_length = 0.4  # Maximum allowed length between consecutive points

for t in range(1, num_steps):
    if t < initial_input_duration:
        states[t, :], _, _ = dynamics(states[t-1, :], initial_input)
    else:
        states[t, :], _, _ = dynamics(states[t-1, :], np.zeros(2))  # Natural evolution with no input

    # Apply maximum length constraint
    for i in range(1, 4):
        delta_y = states[t, i] - states[t, i-1]
        if abs(delta_y) > max_length:
            states[t, i] = states[t, i-1] + np.sign(delta_y) * max_length

# Plot the state evolution
plt.figure()
for i in range(8):
    plt.plot(np.arange(num_steps) * dt, states[:, i], label=f'State {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.legend()
plt.grid()
plt.show()

# Animation of the points
fig, ax = plt.subplots()
ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-1, 1)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Animation of Points 0, 1, 2, 3, 4, 5')

points, = ax.plot([], [], 'bo')  # Blue dots for points
line, = ax.plot([], [], 'b-')  # Blue line connecting points
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

def init():
    points.set_data([], [])
    line.set_data([], [])
    time_text.set_text('')
    return points, line, time_text

def update(frame):
    x_positions = np.linspace(0, 0.8, 6)  # Fixed x positions for points 0 to 5
    y_positions = np.zeros(6)  # Initialize y positions
    y_positions[1:5] = states[frame, :4]  # Extract positions of points 1, 2, 3, 4
    points.set_data(x_positions, y_positions)
    line.set_data(x_positions, y_positions)
    time_text.set_text(f'Time = {frame * dt:.2f} s')
    return points, line, time_text

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=20)

plt.show()