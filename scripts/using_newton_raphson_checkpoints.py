"""
So, the Newton-Raphson method is supposed to be used in place of inverse kinematics to basically allow me to calculate
the joints I need to be at based on some desired position that I want the foot to be in.

Test method:
# define some q_initial
# define some p_desired
# get the current position based on q_initial
"""


import numpy as np
from math import sin, cos, tan, sqrt
from definitions.KinematicChain import KinematicChain

# joint limits
# FL_hip: -1.05 to +1.05 --> midpoint is 0.0
# FL_thigh: -1.57 to 3.49 --> midpoint is 0.96
# FL_calf: -0.84 to -2.72s --> midpoint is -1.78


iter_limit = 40
counter = 0
# q0 = np.array([-1.05, -1.57, -0.84])
q0 = np.array([0.0, -0.01956, -0.1973])
L1 = 0.213
L2 = 0.213
urdf = './models/go2/go2.urdf'

def do_newton_raphson(qcurr, p_desired):
    
    counter = 0

    while counter < iter_limit:
        counter += 1

        t0 = qcurr[0]
        t1 = qcurr[1]
        t2 = qcurr[2]

        # Calculate the Jacobian based on the current joint positions
        # Jacob = np.array([[0,                                                                                                                     -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
        #               [0.0955*tan(t0)*(1/cos(t0)) + L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0) - 0.0955*tan(t0)*(1/cos(t0)), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
        #               [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) - 0.0955*tan(t0)*sin(t0) + 0.0955*(1/cos(t0)),                              L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])
        
        Jacob = np.array([[0,                                                       -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
                      [L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
                      [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) + 0.0955*cos(t0), L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])
        
        # calculate the current foot position based on the current joint positions
        p_base_to_foot_curr, _, _, _ = chain2.fkin([t0, t1, t2])
        p_base_to_hip_curr, _, _, _ = chain3.fkin([t0])
        p_hip_to_foot_curr = p_base_to_foot_curr - p_base_to_hip_curr  

        dist = sqrt((p_hip_to_foot_curr[0] - p_desired[0])**2 + (p_hip_to_foot_curr[1] - p_desired[1])**2 + (p_hip_to_foot_curr[2] - p_desired[2])**2)
        print(f'dist is {dist}') 

        error = p_desired - p_hip_to_foot_curr

        # check the size of the error
        if np.linalg.norm(error) <= 1e-6:
            print(f'Solution successfully found in {counter} iterations')
            return qcurr

        qnext = qcurr + np.linalg.pinv(Jacob) @ error

        qcurr = qnext

        print(f'Current joint angles are {qcurr}')

    if counter >= iter_limit:
        print('Iteration limit reached')
        return [None]

# define the kinematic chains
chain1 = KinematicChain('FR_hip', 'FR_foot', ['FR_thigh_joint', 'FR_calf_joint'], urdf)
chain2 = KinematicChain('base', 'FR_foot', ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'], urdf)
chain3 = KinematicChain('base', 'FR_hip', ['FR_hip_joint'], urdf)

# # define some desired joint positions
# q_desired = np.array([1.05, 1.49, -1.72])

# # feed the current joint positions into the fkin to see what the current foot position is
# p_base_to_foot_curr, _, _, _ = chain2.fkin([q0[0], q0[1], q0[2]])
# p_base_to_hip_curr, _, _, _ = chain3.fkin([q0[0]])
# p_curr = p_base_to_foot_curr - p_base_to_hip_curr

# # feed these desired joint positions into the fkin to see what the foot position should be
# p_base_to_foot, _, _, _ = chain2.fkin([q_desired[0], q_desired[1], q_desired[2]])
# p_base_to_hip, _, _, _ = chain3.fkin([q_desired[0]])
# p_desired = p_base_to_foot - p_base_to_hip


p_curr = np.array([0.05, -0.0955, -0.42097])
p_desired = np.array([0.05, -0.0955, -0.42439])

print(f'Current position: {p_curr}')
print(f'Desired position: {p_desired}')

# get the Euclidean distance between the current and desired positions
distance = sqrt((p_desired[0] - p_curr[0])**2 + (p_desired[1] - p_curr[1])**2 + (p_desired[2] - p_curr[2])**2)
print(f'Distance is: {distance}')

# Max distance (distance between all minimum and maximum joint values) is 0.48 m
# let's try to split this into four pieces? I tried two points with distance 0.12 from each other and it work on NR
# update, didn't work when I tried going between the two joint maxima. Let's do smaller divisions, 0.8?

if distance <= 0.08:
    num_nr = 1

    p_desired_list = [p_desired]

elif distance <= 0.16:
    num_nr = 2

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.5
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.5
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.5

    p_des1 = [x_des1, y_des1, z_des1]
    p_desired_list = [p_des1, p_desired]

elif distance <= 0.24:
    num_nr = 3

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 1/3
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 1/3
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 1/3
    
    x_des2 = p_curr[0] + (p_desired[0] - p_curr[0]) * 2/3
    y_des2 = p_curr[1] + (p_desired[1] - p_curr[1]) * 2/3
    z_des2 = p_curr[2] + (p_desired[2] - p_curr[2]) * 2/3

    p_des1 = [x_des1, y_des1, z_des1]
    p_des2 = [x_des2, y_des2, z_des2]
    p_desired_list = [p_des1, p_des2, p_desired]

elif distance <= 0.32:
    num_nr = 4

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.25
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.25
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.25

    x_des2 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.5
    y_des2 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.5
    z_des2 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.5

    x_des3 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.75
    y_des3 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.75
    z_des3 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.75

    p_des1 = [x_des1, y_des1, z_des1]
    p_des2 = [x_des2, y_des2, z_des2]
    p_des3 = [x_des3, y_des3, z_des3]
    p_desired_list = [p_des1, p_des2, p_des3, p_desired]

elif distance <= 0.40:
    
    num_nr = 5

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.2
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.2
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.2

    x_des2 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.4
    y_des2 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.4
    z_des2 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.4

    x_des3 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.6
    y_des3 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.6
    z_des3 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.6

    x_des4 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.8
    y_des4 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.8
    z_des4 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.8

    p_des1 = [x_des1, y_des1, z_des1]
    p_des2 = [x_des2, y_des2, z_des2]
    p_des3 = [x_des3, y_des3, z_des3]
    p_des4 = [x_des4, y_des4, z_des4]
    p_desired_list = [p_des1, p_des2, p_des3, p_des4, p_desired]

elif distance <= 0.48:
    num_nr = 6

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 1/6
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 1/6
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 1/6

    x_des2 = p_curr[0] + (p_desired[0] - p_curr[0]) * 2/6
    y_des2 = p_curr[1] + (p_desired[1] - p_curr[1]) * 2/6
    z_des2 = p_curr[2] + (p_desired[2] - p_curr[2]) * 2/6

    x_des3 = p_curr[0] + (p_desired[0] - p_curr[0]) * 0.5
    y_des3 = p_curr[1] + (p_desired[1] - p_curr[1]) * 0.5
    z_des3 = p_curr[2] + (p_desired[2] - p_curr[2]) * 0.5

    x_des4 = p_curr[0] + (p_desired[0] - p_curr[0]) * 4/6
    y_des4 = p_curr[1] + (p_desired[1] - p_curr[1]) * 4/6
    z_des4 = p_curr[2] + (p_desired[2] - p_curr[2]) * 4/6

    x_des5 = p_curr[0] + (p_desired[0] - p_curr[0]) * 5/6
    y_des5 = p_curr[1] + (p_desired[1] - p_curr[1]) * 5/6
    z_des5 = p_curr[2] + (p_desired[2] - p_curr[2]) * 5/6

    p_des1 = [x_des1, y_des1, z_des1]
    p_des2 = [x_des2, y_des2, z_des2]
    p_des3 = [x_des3, y_des3, z_des3]
    p_des4 = [x_des4, y_des4, z_des4]
    p_des5 = [x_des5, y_des5, z_des5]
    p_desired_list = [p_des1, p_des2, p_des3, p_des4, p_des5, p_desired]

else:
    num_nr = 9

    x_des1 = p_curr[0] + (p_desired[0] - p_curr[0]) * 1/9
    y_des1 = p_curr[1] + (p_desired[1] - p_curr[1]) * 1/9
    z_des1 = p_curr[2] + (p_desired[2] - p_curr[2]) * 1/9

    x_des2 = p_curr[0] + (p_desired[0] - p_curr[0]) * 2/9
    y_des2 = p_curr[1] + (p_desired[1] - p_curr[1]) * 2/9
    z_des2 = p_curr[2] + (p_desired[2] - p_curr[2]) * 2/9

    x_des3 = p_curr[0] + (p_desired[0] - p_curr[0]) * 3/9
    y_des3 = p_curr[1] + (p_desired[1] - p_curr[1]) * 3/9
    z_des3 = p_curr[2] + (p_desired[2] - p_curr[2]) * 3/9

    x_des4 = p_curr[0] + (p_desired[0] - p_curr[0]) * 4/9
    y_des4 = p_curr[1] + (p_desired[1] - p_curr[1]) * 4/9
    z_des4 = p_curr[2] + (p_desired[2] - p_curr[2]) * 4/9

    x_des5 = p_curr[0] + (p_desired[0] - p_curr[0]) * 5/9
    y_des5 = p_curr[1] + (p_desired[1] - p_curr[1]) * 5/9
    z_des5 = p_curr[2] + (p_desired[2] - p_curr[2]) * 5/9

    x_des6 = p_curr[0] + (p_desired[0] - p_curr[0]) * 6/9
    y_des6 = p_curr[1] + (p_desired[1] - p_curr[1]) * 6/9
    z_des6 = p_curr[2] + (p_desired[2] - p_curr[2]) * 6/9

    x_des7 = p_curr[0] + (p_desired[0] - p_curr[0]) * 7/9
    y_des7 = p_curr[1] + (p_desired[1] - p_curr[1]) * 7/9
    z_des7 = p_curr[2] + (p_desired[2] - p_curr[2]) * 7/9

    x_des8 = p_curr[0] + (p_desired[0] - p_curr[0]) * 8/9
    y_des8 = p_curr[1] + (p_desired[1] - p_curr[1]) * 8/9
    z_des8 = p_curr[2] + (p_desired[2] - p_curr[2]) * 8/9

    p_des1 = [x_des1, y_des1, z_des1]
    p_des2 = [x_des2, y_des2, z_des2]
    p_des3 = [x_des3, y_des3, z_des3]
    p_des4 = [x_des4, y_des4, z_des4]
    p_des5 = [x_des5, y_des5, z_des5]
    p_des6 = [x_des6, y_des6, z_des6]
    p_des7 = [x_des7, y_des7, z_des7]
    p_des8 = [x_des8, y_des8, z_des8]
    p_desired_list = [p_des1, p_des2, p_des3, p_des4, p_des5, p_des6, p_des7, p_des8, p_desired]

for n in range(num_nr):
    p_des = p_desired_list[n]
    joint_angles = do_newton_raphson(q0, p_des)
    q0 = joint_angles

    if None in q0:
        print(f'We flopped during iteration {n}, chief :/')
        break

print(f'Joint angles found from Newton-Raphson method: {joint_angles}')