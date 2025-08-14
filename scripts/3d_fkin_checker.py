import numpy as np
from math import acos, atan2, sqrt, sin, cos, tan

from definitions.KinematicChain import KinematicChain

from hw5code.TransformHelpers import *

import matplotlib.pyplot as plt


# we start by first defining the link lengths
L1 = 0.213
L2 = 0.213
theta0 = 0.1
theta1 = -0.8
theta2 = -0.9

urdf = './models/go2/go2.urdf'

chain1 = KinematicChain('FL_hip', 'FL_foot', ['FL_thigh_joint', 'FL_calf_joint'], urdf)

chain2 = KinematicChain('base', 'FL_foot', ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'], urdf)

chain3 = KinematicChain('base', 'FL_hip', ['FL_hip_joint'], urdf)

p_base_to_foot, _, _, _ = chain2.fkin([theta0, theta1, theta2])
p_base_to_hip, _, _, _ = chain3.fkin([theta0])

# Subtract to get p_hip_to_foot
p_hip_to_foot = p_base_to_foot - p_base_to_hip

# print('check positions')
# print(f'Position base to foot: {p_base_to_foot}')
# print(f'Position base to hip: {p_base_to_hip}')
print(f'Position hip to foot: {p_hip_to_foot}')

### FKIN checker
x = -L1*sin(theta1) - L2*sin(theta1 + theta2)
# print(L1*sin(theta1))
# print(L2*sin(theta2 - theta1))
y = 0.0955/cos(theta0) + (L1*cos(theta1) + L2*cos(theta1 + theta2) - 0.0955*tan(theta0))*sin(theta0)
# print(0.0955/cos(theta0))
# print(sin(theta0))
# print((-L1*cos(theta1) - L2*cos(theta2 - theta1)))
# print( + 0.0955*tan(theta0))
z = (-L1*cos(theta1) - L2*cos(theta2 + theta1) + 0.0955*tan(theta0))*cos(theta0)

print(f'Calc Position: {[round(x,8), round(y,8), round(z,8)]}')


# build one chain from the base to the foot. Build another chain from the base to the hip
# subtract them to get the actual fkin from the hip to the foot

