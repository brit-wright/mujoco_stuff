#!/usr/bin/env python3.10

import numpy as np
from math import acos, atan2, sqrt, sin, cos

from definitions.KinematicChain import KinematicChain

from hw5code.TransformHelpers import *

import matplotlib.pyplot as plt


# we start by first defining the link lengths
L1 = 0.213
L2 = 0.213

# define the base position of the leg in [x, z] coordinates
ORIGIN = [0.0, 0.0]

LEN_STRETCHED = L1 + L2

# define the function to calculate theta2
def find_theta2(r):
    
    cos_theta2 = (r**2 - L2**2 - L1**2)/(2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -acos(cos_theta2)

    return theta2

def find_theta1(theta2, z, x):
    # theta1 = - atan2(L2*sin(theta2), L1 + L2*cos(theta2))
    # theta1 = atan2(z, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))
    # theta1 = atan2(x, z) - atan2(L2*sin(theta2), L1+L2*cos(theta2))
    alpha = atan2(x, z)
    beta = atan2(L2*abs(sin(theta2)), -(L1+L2*cos(theta2)))
    theta1 = alpha - beta
    print(f'Parameters: alpha: {alpha}, beta: {beta}, theta1: {theta1}')
    return theta1

def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def check_point(point):
    # we essentially check that the point is not equal to the origin and that the distance of the point from
    # the origin is not greater than the length of the stretched leg
    
    if point[0] == ORIGIN[0] and point[1] == ORIGIN[1]:
        return False
    
    else:
        dist1 = calculate_distance(ORIGIN, point)
        
        if dist1 > LEN_STRETCHED:
            return False
        
    return True
    
# get some points :/

# define a radius
r = 0.2

thetas = np.arange(0, 2*np.pi, 0.1)

# define the kinematic chain for the arm
urdf = './models/go2/go2.urdf'
kin_chain = KinematicChain('FL_hip', 'FL_foot', ['FL_thigh_joint', 'FL_calf_joint'], urdf)

originalsx = []
originalsz = []
ikinsx = []
ikinsz = []

r_vals = np.arange(0.0, 0.5, 0.05)
# r_vals = [0.2]

for r in r_vals:
    for ang in thetas:
        x_val = r*cos(ang)
        z_val = r*sin(ang)
        pt = [x_val, z_val]

        if check_point(pt) == True:

            # using the end point, get the associated joint angles (theta1, theta2)
            r_dist = calculate_distance(ORIGIN, pt)

            # print(r_dist)

            thet2 = find_theta2(r_dist)
            thet1 = find_theta1(thet2, z_val, x_val)

            p, _, _, _ = kin_chain.fkin([thet1, thet2])
            
            # print([thet1, thet2])
            # print(p)

            originalsx.append(pt[0])
            originalsz.append(pt[1])

            ikinsx.append(p[0])
            ikinsz.append(p[2])

plt.plot(originalsx, originalsz, 'k')
plt.scatter(ikinsx, ikinsz)
plt.show()

#     # calculate the forward kinematics and see if I'll get the original pt
