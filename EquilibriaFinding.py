import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import root

# System constants
alpha = 128*0.25
c = 0.1
dstep = 1e-3
d = 0.2
m = [0.1, 0.2, 0.1, 0.2]
n_points = 4  # Number of surface points
max_movement=0.02

#lets use the hypothesis of having a surface lenght of 1 meter max
max_lenght = 1.5

# Define Lij
def Lij(iter, i, xx):
    return np.sqrt((xx[iter] - xx[i])**2 + ((d * (iter - i))**2))


def total_lenght(xx):
    lenght=0
    
    for i in range(n_points-1):
        lenght += Lij(i,i+1,xx)

    return lenght

# Define sum function
def sum_func(xx, iter):
    temp = 0
    for i in range(4):
        if i != iter:
            temp += (xx[iter] - xx[i]) / (Lij(iter, i, xx) * (Lij(iter, i, xx)**2 - (xx[iter] - xx[i])**2))
    return temp

# Define dynamics with maximum movement constraint
def dynamics(xx):
    xxs = np.zeros_like(xx)

    for i in range(4):
        # Update y-coordinates
        proposed_y = xx[i] + dstep * xx[i + 4]
        displacement = proposed_y - xx[i]
        tot = total_lenght(xx)
        if  tot > max_lenght:
            print('max lenght exceeded:',tot)
            displacement =- np.sign(displacement) * (max_lenght-tot)  # Limit to max_movement
            proposed_y = xx[i] + displacement

        xxs[i] = proposed_y
        
        # Update velocities
        xxs[i + 4] = xx[i + 4] + dstep * (1 / m[i]) * (-alpha * sum_func(xx, i) - c * xx[i + 4])
    return xxs


# Equilibrium equations
def equilibrium_eqs(xx):
    eqs = []
    for i in range(4):
        eqs.append(-alpha * sum_func(xx, i))  # Set sum to 0 for equilibrium
    return eqs

# Solve for equilibrium
initial_guess = np.random.uniform(-1.0, 1.0, size=4)
solution = root(equilibrium_eqs, initial_guess[:4])

if not solution.success:
    raise ValueError("Failed to find equilibrium.")
equilibrium = solution.x

print(initial_guess)
print(solution.x)

# Initial condition (y-coordinates + velocities)
initial_condition = np.hstack((np.random.uniform(-0.05, 0.05, 4), np.random.uniform(-0.05, 0.05, 4)))

# Time simulation parameters
n_steps = 300

# Simulate system evolution
trajectory = [initial_condition]
for _ in range(n_steps):
    trajectory.append(dynamics(trajectory[-1]))
trajectory = np.array(trajectory)

# Extract y-coordinates for plotting
y_coords = trajectory[:, :4]

# Fixed x-coordinates for surface points
x_coords = np.arange(n_points)

# Create dynamic plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2, label="Surface")
ax.legend()
ax.set_xlim(-1, n_points)  # Keep x-coordinates fixed
ax.set_ylim(-2, 2)        # Adjust based on expected y-coordinate range
ax.set_title("Dynamic Surface Evolution")
ax.set_xlabel("x")
ax.set_ylabel("y")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x_coords, y_coords[frame])
    return line,

ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=50)

plt.show()