# plotting Bezier curves in 3D
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

# define the control points to be used to characterize the
# curve

deg = 5
p = []
rang = 1

# create a function to calculate bezier coefficients
def get_coeffs(n, i):
    
    coeff = factorial(n) / (factorial(i) * factorial(n - i))
    # print(f'coefficient is: {coeff}')
    return coeff

# define the bezier function
def create_bezier(points, time):
    sum = 0
    n = len(points) -1
    z_list = []
    for idx in range(n+1): # from 0 to 5
        term = get_coeffs(n, idx) * (1 - time)**(n-idx) * time**idx * points[idx]
        sum += term
        z_list.append(term[-1])

    return sum, z_list

# create control points
# for i in range(deg): # creates deg control points (0 to 5)
#     p_rand = rang * np.random.rand(3, 1)
#     p.append(p_rand)

p = [np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.0]), np.array([0.4, 0.0, 0.8]),
     np.array([0.4, 0.0, 0.0]), np.array([0.8, 0.0, 0.0])]

# get the values for the bezier curve for each timestep
vals = []
dt = 0.005
time_points = np.arange(0, 1+dt, dt)
max_z_list = []

for t in time_points:
    val, z_list = create_bezier(p, t)
    vals.append(val)
    max_z_list.append(max(z_list))

max_z = max(max_z_list)
print(f'highest point: {max_z}')

des_amplitude = 0.8
fact = des_amplitude/max_z

# plotting
ax = plt.axes(projection='3d')

x_vals = [val[0] for val in p]
y_vals = [val[1] for val in p]
z_vals = [val[2] for val in p]

x_vals_bez = [val[0] for val in vals]
y_vals_bez = [val[1] for val in vals]
z_vals_bez = [fact*val[2] for val in vals]

# plot the control points
ax.scatter3D(x_vals, y_vals, z_vals)

# plot the bezier function points
ax.plot3D(x_vals_bez, y_vals_bez, z_vals_bez)

# plot the connections between the control points
for i in range(len(p)-1):
    x_vals = [p[i][0], p[i+1][0]]
    y_vals = [p[i][1], p[i+1][1]]
    z_vals = [p[i][2], p[i+1][2]]
    ax.plot3D(x_vals, y_vals, z_vals, 'g-')

plt.show()

