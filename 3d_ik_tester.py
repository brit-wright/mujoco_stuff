#!/usr/bin/env python3.10

import numpy as np
from math import acos, atan2, sqrt, sin, cos

from definitions.KinematicChain import KinematicChain

from hw5code.TransformHelpers import *

import matplotlib.pyplot as plt


# we start by first defining the link lengths
L1 = 0.213
L2 = 0.213

# define the base position of the leg in [x, y, z] coordinates
ORIGIN = [0.0, 0.0, 0.0]

LEN_STRETCHED = L1 + L2

# define the function to calculate theta2
def find_theta0(y, z):

    theta0 = atan2(y, abs(z))
    return theta0

def find_theta2(d):

    cos_theta2 = (d**2 - L1**2 - L2**2)/(2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -acos(cos_theta2)
    
    return theta2

def find_theta1(theta2, proj, x):
    
    alpha = atan2(x, proj)
    beta = atan2(L2*abs(sin(theta2)), -(L1+L2*cos(theta2)))
    theta1 = alpha - beta
    print(f'Parameters: alpha: {alpha}, beta: {beta}, theta1: {theta1}')
    return theta1

def calculate_proj(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def check_point(point):
    # we essentially check that the point is not equal to the origin and that the distance of the point from
    # the origin is not greater than the length of the stretched leg
    
    if point == ORIGIN:
        return False
    
    else:
        dist1 = calculate_distance(ORIGIN, point)
        
        if dist1 > LEN_STRETCHED:
            return False
        
    return True
    
# # define the angles
# thetas = np.arange(1.0, np.pi, 0.1)
# phis = np.arange(1.0, np.pi, 0.1)

# # define the kinematic chain for the arm
urdf = './models/go2/go2.urdf'
# kin_chain3 = KinematicChain('FL_hip', 'FL_foot', ['FL_thigh_joint', 'FL_calf_joint'], urdf)
# originalsx = []
# originalsy = []
# originalsz = []
# ikinsx = []
# ikinsy = []
# ikinsz = []

# # define the test radii
# # r_vals = np.arange(0.0, 0.5, 0.05)
# r_vals = [0.2]


# # new plan
# # adjust method to do sampling

# # get theta values theta0, theta1, theta2
# theta0 = 1.0
# theta1 = 0.0
# theta2 = -0.84

# # apply fkin to get what the end position should be
# p3, _, _, _ = kin_chain3.fkin([theta1, theta2])

# print(f'Position of p3: {p3}')

# toe_offset = np.array([0, 0.0955, -0.426])

# # adjusted input toe position relative to "vertical"
# p3_adjusted = p3 - toe_offset

# print(f'Adjusted Position of p3: {p3_adjusted}')

# # apply fkin given this position to see if the correct angles are returned
# proj_dist = sqrt(p3_adjusted[1]**2 + p3_adjusted[2]**2)
# d = sqrt(proj_dist**2 + p3_adjusted[0]**2)

# # send these to the ikin functions
# thet0 = find_theta0(proj_dist, p3_adjusted[1], p3_adjusted[2])
# thet2 = find_theta2(d)
# thet1 = find_theta1(thet2, proj_dist, p3_adjusted[0])

# print(f'Given angles: {theta0, theta1, theta2}')
# print(f'IKIN result: {thet0, thet1, thet2}')

# p3 = kin_chain3.fkin([0.0, 0.0])[0]
# print(f'Foot position with zero angles (p3): {p3}')


# we're calculating theta0, theta1, theta2

# to calculate these thetas, we need to know the position of the foot in the hip's coordinate system.
# build the kinematic chain from the hip to the foot
kin_chain1 = KinematicChain('FL_thigh', 'FL_foot', ['FL_calf_joint'], urdf)
kin_chain2 = KinematicChain('FL_hip', 'FL_thigh', ['FL_thigh_joint'], urdf)

# next let's look at the position of the foot with respect to the hip when all the angles are zeros
p_foot_thigh, _, _, _ = kin_chain1.fkin([0.0])
_, R_thigh_hip, _, _ = kin_chain2.fkin([0.0])

print(f'p is {p_foot_thigh}')

# at 0 degrees, p = [0.0, 0.0955, -0.426]
theta_default = atan2(0.0955, 0.426)
print(f'Theta_default: {theta_default}')

# this means that at 0 degrees, the foot is not in line with the thigh. 

# let's define some angles we wanna pass in:
theta0 = 0.5
theta1 = 0.0
theta2 = -0.84

p_foot_thigh_cleaned = p_foot_thigh - [0.0, 0.0955, 0.0]
p_foot_hip = R_thigh_hip.T @ p_foot_thigh_cleaned

theta0_ikin = find_theta0(p_foot_hip[1], p_foot_hip[2])
print(f'IKIN theta0: {theta0_ikin}')
theta0_ikin = theta0_ikin + theta_default
print(f'IKIN theta0: {theta0_ikin}')



