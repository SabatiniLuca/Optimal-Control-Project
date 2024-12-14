#
# Course Project #2: Optimal Control of an Actuated Flexible Surface
# Group 17: Faedo Nicolò, Paltrinieri Mirco, Sabatini Luca
# Bologna, December 2024
#
# rootfinding file

import numpy as np 
import matplotlib.pyplot as plt
import dynamics as dyn
from matplotlib.animation import FuncAnimation 
from scipy.interpolate import CubicSpline


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


#parameters
max_iters = 100
stepsize = 0.23
tol = 1

d = dyn.d
dt = dyn.dt
n_points = dyn.n_points
ns = dyn.ns
ni = dyn.ni

Convergence_flag = False
randInit = False


#initializations
if not randInit: xx_init = np.array([0, 0.3, -0.3, 0.3, 0,0,0,0])
else :xx_init = np.random.uniform(-1,1,8)
uu_init = np.array([0,0])


#plot initializations
states = np.zeros((ns, max_iters))
inputs = np.zeros((ni, max_iters))
gradients = np.zeros ((ns,ns,max_iters))
# print('states shape:', states.shape)
# print('input shape', inputs.shape)
norms = np.zeros((1,max_iters))


#for look root finding process
for k in range(max_iters):
    if k==0 :
        xx_updated = xx_init
        uu_updated = uu_init

    xx, dfx, dfu = dyn.dynamics(xx_updated,uu_updated)

    direction =  -np.linalg.solve(dfx, xx)

    xx_updated = xx + stepsize * direction

    states[:,k] = xx_updated
    inputs[:,k] = uu_updated
    gradients[:,:,k] = dfx

    norm = np.linalg.norm(xx_updated)

    norms[:,k]= norm

    if norm < tol:
        print('convergence obtained with',k,' iterations /n converngence value:', norm)
        Convergence_flag = True
        break




#animation and plots
fig , ax = plt.subplots()
ax.set_xlim(-0.1, d * (n_points ))
ax.set_ylim(-10, 10)
line, = ax.plot([], [], 'b', lw=2, label=f"Initial state: {xx_init}")
ax.legend(loc="upper right")

# Animation update function
def update(frame):
    
    # Update positions
    state = states[:, frame]
    
    # Extract positions and use cubic spline to interpolate
    z = state[:4]  # Assuming the first 4 states are positions
    x = np.array([0, d, 2*d, 3*d])
    spline = CubicSpline(x, z)
    x_smooth = np.linspace(0, 3*d, 100)
    y_smooth = spline(x_smooth)
    
    # Update the line data
    line.set_data(x_smooth, y_smooth)
    
    return line,
# Create animation
ani = FuncAnimation(fig, update, frames=max_iters, interval=50, blit=True)
# plt.show()


# Subplots for states over time
time = np.arange(max_iters) * dt  # Assuming each iteration corresponds to a time step

fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle("States and Norm Over Time")

#convergence time 
if Convergence_flag: convergence_time = k*dt

# First four states (positions)
for i in range(4):
    ax1.plot(time, states[i, :], label=f"Position: z{i+1}")
if Convergence_flag: ax1.axvline(convergence_time, color='r', linestyle='--', label=f"Convergence at t={convergence_time:.2f}s")
ax1.set_title("First Four States (Positions)")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("State Value")
ax1.legend(loc="upper right")
ax1.grid(True)

# Last four states (velocities)
for i in range(4, 8):
    ax2.plot(time, states[i, :], label=f"Velocity: z'{i+1}")
if Convergence_flag: ax2.axvline(convergence_time, color='r', linestyle='--', label=f"Convergence at t={convergence_time:.2f}s")
ax2.set_title("Last Four States (Velocities)")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("State Value")
ax2.legend(loc="upper right")
ax2.grid(True)

ax3.plot(time, norms[0, :], label="Norm")
if Convergence_flag:
    ax3.axvline(convergence_time, color='r', linestyle='--', label=f"Convergence at t={convergence_time:.2f}s")
ax3.set_title("Norm of the States Over Time")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Norm Value")
ax3.legend(loc="upper right")
ax3.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

