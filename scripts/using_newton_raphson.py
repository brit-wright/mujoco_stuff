import numpy as np
from math import sin, cos, tan
from definitions.KinematicChain import KinematicChain

# joint limits
# FL_hip: -1.05 to +1.05 --> midpoint is 0.0
# FL_thigh: -1.57 to 3.49 --> midpoint is 0.96
# FL_calf: -0.84 to -2.72s --> midpoint is -1.78


iter_limit = 40
counter = 0
# q0 = np.array([0.0, 0.96, -1.78])
q0 = np.array([0.0, -0.01956, -0.1973])
qlast = q0
L1 = 0.213
L2 = 0.213

xd = np.array([0.364, 0.107, -0.111])

urdf = './models/go2/go2.urdf'

chain1 = KinematicChain('FL_hip', 'FL_foot', ['FL_thigh_joint', 'FL_calf_joint'], urdf)

chain2 = KinematicChain('base', 'FL_foot', ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'], urdf)

chain3 = KinematicChain('base', 'FL_hip', ['FL_hip_joint'], urdf)


# define some desired joint positions
# q_desired = np.array([-1.05, 3.49, -0.8])
q_desired = np.array([1.05, 1.49, -1.72])

# feed these desired joint positions into the fkin to see what the foot position should be
p_base_to_foot, _, _, _ = chain2.fkin([q_desired[0], q_desired[1], q_desired[2]])
p_base_to_hip, _, _, _ = chain3.fkin([q_desired[0]])
p_desired = p_base_to_foot - p_base_to_hip


p_desired = np.array([0.05, -0.0955, -0.424])

while counter <  iter_limit:

    counter += 1
    
    t0 = qlast[0]
    t1 = qlast[1]
    t2 = qlast[2]

    # calculate the jacobian based on the last joint values found
    Jacob = np.array([[0,                                                                                                                     -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
                      [0.0955*tan(t0)*(1/cos(t0)) + L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0) - 0.0955*tan(t0)*(1/cos(t0)), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
                      [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) - 0.0955*tan(t0)*sin(t0) + 0.0955*(1/cos(t0)),                              L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])

    # calculate the current foot position (using fkin)
    p_base_to_foot_curr, _, _, _ = chain2.fkin([t0, t1, t2])
    p_base_to_hip_curr, _, _, _ = chain3.fkin([t0])

    p_hip_to_foot_curr = p_base_to_foot_curr - p_base_to_hip_curr       

    error = p_desired - p_hip_to_foot_curr
    print(f'Error is: {error}')

    # terminate if the error is small
    if np.linalg.norm(error) <= 1e-6:
        print(f'Solution successfully found in {counter} iterations')
        break

    # q_curr = qlast + np.linalg.pinv(Jacob) @ error
    q_curr = qlast + np.linalg.pinv(Jacob) @ error

    qlast = q_curr

    print(f'Current joint angles: {q_curr}')

if np.linalg.norm(error) <= 1e-6:
    
    # print the joint angles found from newton-raphson
    print(f'Joint Angles from Newton-Raphson: {q_curr}')

else:
    print('iteration limit reached')


# while counter < iter_limit:

#     t0 = qlast[0]
#     t1 = qlast[1]
#     t2 = qlast[2]

#     p_base_to_foot, _, _, _ = chain2.fkin([t0, t1, t2])
#     p_base_to_hip, _, _, _ = chain3.fkin([t0])

#     p_hip_to_foot = p_base_to_foot - p_base_to_hip

#     Jacob = np.array([[0,                                                                                                                     -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
#                       [0.0955*tan(t0)*(1/cos(t0)) + L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0) - 0.0955*tan(t0)*(1/cos(t0)), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
#                       [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) - 0.0955*tan(t0)*sin(t0) + 0.0955*(1/cos(t0)),                              L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])

#     # error is desired minus fkin
#     error = xd - p_hip_to_foot
#     print(f'Error is: {error}')

#     if np.linalg.norm(error) <= 0.0001:
#         break

#     q_curr = qlast + np.linalg.pinv(Jacob) @ error

#     qlast = q_curr

#     print(f'Q-curr is: {q_curr}')

# if np.linalg.norm(error) <= 0.0001:
#     print('exited successfully')
# else:
#     print('iteration limit reached')