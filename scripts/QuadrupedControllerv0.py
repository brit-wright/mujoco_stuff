import numpy as np
import time
import yaml
from math import sin, cos, sqrt
import mujoco

from definitions.go2_definitions import Mujoco_IDX_go2
from definitions.KinematicChain import KinematicChain
from definitions.NewtonRaphson import NewtonRaphson
from definitions.Bezier2D import Bezier2D
from definitions.Bezier3D import Bezier3D

"""
Let's do a little pre-amble to justify having this file in the first place. 

I think the whole idea is that I basically wanna tell the robot to do certain movements now?

I currently have the trajectory class which defines desired positions and does ikin on them to get the 
necessary joint angles

Like for example, with the standing, I tell the robot that I want it to retain it's initial position
from the yaml file


Things I want in the init section
1. the yaml file that defines configuration information about the robot

"""

SEMI_MIN = 0.00194
SEMI_MAJ = 0.0054

class QuadrupedController:

    def __init__(self, data, mj_idx, q0_joint):

        self.data = data
        self.mj_idx = mj_idx

        # initialize the joint indices
        self.fl = [0, 1, 2]
        self.fr = [3, 4, 5]
        self.rl = [6, 7, 8]
        self.rr = [9, 10, 11]

        self.x_list = []
        self.t_list = []
        self.z_list = []
        
        # define the initial joint positions
        q = data.qpos
        self.q_joints = q[mj_idx.q_joint_idx]
        self.qstable = self.q_joints
        self.qcurr = self.qstable

        self.q0_joint = q0_joint

        self.cycle = 0

        self.urdf = './models/go2/go2.urdf'

        self.cycle_len = 0.4

        # set up the kinematic chains
        # chains from base to foot
        self.chain_base_foot_fl = KinematicChain('base', 'FL_foot', self.jointnames()[0:3], self.urdf)
        self.chain_base_foot_fr = KinematicChain('base', 'FR_foot', self.jointnames()[3:6], self.urdf)
        self.chain_base_foot_rl = KinematicChain('base', 'RL_foot', self.jointnames()[6:9], self.urdf)
        self.chain_base_foot_rr = KinematicChain('base', 'RR_foot', self.jointnames()[9:12], self.urdf)


        # chains from base to hip
        self.chain_base_hip_fl = KinematicChain('base', 'FL_hip', self.jointnames()[0:1], self.urdf)
        self.chain_base_hip_fr = KinematicChain('base', 'FR_hip', self.jointnames()[3:4], self.urdf)
        self.chain_base_hip_rl = KinematicChain('base', 'RL_hip', self.jointnames()[6:7], self.urdf)
        self.chain_base_hip_rr = KinematicChain('base', 'RR_hip', self.jointnames()[9:10], self.urdf)


        # chains from base to thigh
        self.chain_base_thigh_fl = KinematicChain('base', 'FL_thigh', self.jointnames()[0:2], self.urdf)
        self.chain_base_thigh_fr = KinematicChain('base', 'FR_thigh', self.jointnames()[3:5], self.urdf)
        self.chain_base_thigh_rl = KinematicChain('base', 'RL_thigh', self.jointnames()[6:8], self.urdf)
        self.chain_base_thigh_rr = KinematicChain('base', 'RR_thigh', self.jointnames()[9:11], self.urdf)

        # get the initial position of the foot
        self.p0_base_foot_fl = self.chain_base_foot_fl.fkin(self.q_joints[self.fl[0]: self.fl[-1]+1])[0]
        self.p0_base_hip_fl = self.chain_base_hip_fl.fkin(self.q_joints[self.fl[0]:self.fl[1]])[0]
        self.p0_base_thigh_fl = self.chain_base_thigh_fl.fkin(self.q_joints[self.fl[0]: self.fl[-1]])[0]

        self.p0_foot_fl = self.p0_base_foot_fl - self.p0_base_hip_fl
        self.p0_thigh_fl = self.p0_base_thigh_fl - self.p0_base_hip_fl


        self.p0_base_foot_fr = self.chain_base_foot_fr.fkin(self.q_joints[self.fr[0]: self.fr[-1]+1])[0]
        self.p0_base_hip_fr = self.chain_base_hip_fr.fkin(self.q_joints[self.fr[0]:self.fr[1]])[0]
        self.p0_base_thigh_fr = self.chain_base_thigh_fr.fkin(self.q_joints[self.fr[0]: self.fr[-1]])[0]

        self.p0_foot_fr = self.p0_base_foot_fr - self.p0_base_hip_fr
        self.p0_thigh_fr = self.p0_base_thigh_fr - self.p0_base_hip_fr


        self.p0_base_foot_rl = self.chain_base_foot_rl.fkin(self.q_joints[self.rl[0]: self.rl[-1]+1])[0]
        self.p0_base_hip_rl = self.chain_base_hip_rl.fkin(self.q_joints[self.rl[0]:self.rl[1]])[0]
        self.p0_base_thigh_rl = self.chain_base_thigh_rl.fkin(self.q_joints[self.rl[0]: self.rl[-1]])[0]

        self.p0_foot_rl = self.p0_base_foot_rl - self.p0_base_hip_rl
        self.p0_thigh_rl = self.p0_base_thigh_rl - self.p0_base_hip_rl


        self.p0_base_foot_rr = self.chain_base_foot_rr.fkin(self.q_joints[self.rr[0]: self.rr[-1]+1])[0]
        self.p0_base_hip_rr = self.chain_base_hip_rr.fkin(self.q_joints[self.rr[0]:self.rr[1]])[0]
        self.p0_base_thigh_rr = self.chain_base_thigh_rr.fkin(self.q_joints[self.rr[0]: self.rr[-1]])[0]

        self.p0_foot_rr = self.p0_base_foot_rr - self.p0_base_hip_rr
        self.p0_thigh_rr = self.p0_base_thigh_rr - self.p0_base_hip_rr


        self.l_thigh = 0.213
        self.l_calf = 0.213

        # Define the stance timings from self.cycle_len
        self.single_stance_time = 3 * self.cycle_len/8
        self.double_stance_time = self.cycle_len/8

        self.rot_speed = 0.8
        self.delta_theta = self.rot_speed * self.single_stance_time

        self.r = sqrt((0.1934)**2 + (0.0955+0.0465)**2)

    # jointnames helper function
    def jointnames(self):
        # 12 joints
        return['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    def stabilize_forward(self, t, T, z_comm, vx): # assume deltax is zero
        
        self.deltx = vx * self.single_stance_time

        # new stance positioning
        new_deltx = (self.deltx - (self.deltx * (self.double_stance_time/self.cycle_len))) * 1/2
        
        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0] + new_deltx


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0] - new_deltx

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0] - new_deltx

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0] + new_deltx

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_x = (1 - alph) * p0_FLx + alph * pf_FLx
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [pdes_x, self.p0_foot_fl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_x = (1 - alph) * p0_FRx + alph * pf_FRx
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [pdes_x, self.p0_foot_fr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_x = (1 - alph) * p0_RLx + alph * pf_RLx
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [pdes_x, self.p0_foot_rl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_x = (1 - alph) * p0_RRx + alph * pf_RRx
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [pdes_x, self.p0_foot_rr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable
    

    def stabilize_reverse(self, t, T, z_comm, vx): # assume deltax is zero
        
        self.deltx = vx * self.single_stance_time

        # new stance positioning
        new_deltx = (self.deltx - (self.deltx * (self.double_stance_time/self.cycle_len))) * 1/2
        
        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0] - new_deltx


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0] + new_deltx

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0] + new_deltx

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0] - new_deltx

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_x = (1 - alph) * p0_FLx + alph * pf_FLx
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [pdes_x, self.p0_foot_fl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_x = (1 - alph) * p0_FRx + alph * pf_FRx
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [pdes_x, self.p0_foot_fr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_x = (1 - alph) * p0_RLx + alph * pf_RLx
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [pdes_x, self.p0_foot_rl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_x = (1 - alph) * p0_RRx + alph * pf_RRx
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [pdes_x, self.p0_foot_rr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable
    
    def stabilize_left_sidestep(self, t, T, z_comm, vy): # assume deltax is zero
        
        self.delty = vy * self.single_stance_time

        # new stance positioning
        
        p0_FLy = self.p0_foot_fl[1]
        pf_FLy = self.p0_thigh_fl[1] + self.delty/2


        p0_FRy = self.p0_foot_fr[1]
        pf_FRy = self.p0_thigh_fr[1] - self.delty/2

        p0_RLy = self.p0_foot_rl[1]
        pf_RLy = self.p0_thigh_rl[1] - self.delty/2

        p0_RRy = self.p0_foot_rr[1]
        pf_RRy = self.p0_thigh_rr[1] + self.delty/2

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_y = (1 - alph) * p0_FLy + alph * pf_FLy
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [self.p0_foot_fl[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_y = (1 - alph) * p0_FRy + alph * pf_FRy
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [self.p0_foot_fr[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_y = (1 - alph) * p0_RLy + alph * pf_RLy
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [self.p0_foot_rl[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_y = (1 - alph) * p0_RRy + alph * pf_RRy
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [self.p0_foot_rr[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable
    
    def stabilize_right_sidestep(self, t, T, z_comm, vy):

        # lowkey crappy, need to fix :/
        self.delty = vy * self.single_stance_time

        # new stance positioning
        
        p0_FLy = self.p0_foot_fl[1]
        pf_FLy = self.p0_thigh_fl[1] - self.delty/2


        p0_FRy = self.p0_foot_fr[1]
        pf_FRy = self.p0_thigh_fr[1] + self.delty/2

        p0_RLy = self.p0_foot_rl[1]
        pf_RLy = self.p0_thigh_rl[1] + self.delty/2

        p0_RRy = self.p0_foot_rr[1]
        pf_RRy = self.p0_thigh_rr[1] - self.delty/2

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_y = (1 - alph) * p0_FLy + alph * pf_FLy
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [self.p0_foot_fl[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_y = (1 - alph) * p0_FRy + alph * pf_FRy
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [self.p0_foot_fr[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_y = (1 - alph) * p0_RLy + alph * pf_RLy
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [self.p0_foot_rl[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_y = (1 - alph) * p0_RRy + alph * pf_RRy
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [self.p0_foot_rr[0], pdes_y, pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable

    def stabilize_turn_clockwise(self, t, T, z_comm):

        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0]


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0]

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0]

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0]

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_x = (1 - alph) * p0_FLx + alph * pf_FLx
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [pdes_x, self.p0_foot_fl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_x = (1 - alph) * p0_FRx + alph * pf_FRx
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [pdes_x, self.p0_foot_fr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_x = (1 - alph) * p0_RLx + alph * pf_RLx
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [pdes_x, self.p0_foot_rl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_x = (1 - alph) * p0_RRx + alph * pf_RRx
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [pdes_x, self.p0_foot_rr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable

    def stabilize_turn_counterclockwise(self, t, T, z_comm):

        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0]


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0]

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0]

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0]

        # the desired leg z-position is just the negative of the z command
        p0_FLz = self.p0_foot_fl[2]
        pf_FLz = -z_comm

        p0_FRz = self.p0_foot_fr[2]
        pf_FRz = -z_comm

        p0_RLz = self.p0_foot_rl[2]
        pf_RLz = -z_comm

        p0_RRz = self.p0_foot_rr[2]
        pf_RRz = -z_comm

        # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

        alph = t/T

        pdes_x = (1 - alph) * p0_FLx + alph * pf_FLx
        pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
        pcurr_FL = [pdes_x, self.p0_foot_fl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


        pdes_x = (1 - alph) * p0_FRx + alph * pf_FRx
        pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
        pcurr_FR = [pdes_x, self.p0_foot_fr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

        pdes_x = (1 - alph) * p0_RLx + alph * pf_RLx
        pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
        pcurr_RL = [pdes_x, self.p0_foot_rl[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
        pdes_x = (1 - alph) * p0_RRx + alph * pf_RRx
        pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
        pcurr_RR = [pdes_x, self.p0_foot_rr[1], pdes_z]

        # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
        theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
        # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

        self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
                        theta_FR[0], theta_FR[1], theta_FR[2],
                        theta_RL[0], theta_RL[1], theta_RL[2],
                        theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.qcurr = self.qstable
        
        return self.qstable

    def choose_gait_start(self):
        self.p_stable_base_foot_fl = self.chain_base_foot_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]+1])[0]
        self.p_stable_base_foot_fr = self.chain_base_foot_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]+1])[0]
        self.p_stable_base_foot_rl = self.chain_base_foot_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]+1])[0]
        self.p_stable_base_foot_rr = self.chain_base_foot_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]+1])[0]

        self.p_stable_base_hip_fl = self.chain_base_hip_fl.fkin(self.qstable[self.fl[0]: self.fl[1]])[0]
        self.p_stable_base_hip_fr = self.chain_base_hip_fr.fkin(self.qstable[self.fr[0]: self.fr[1]])[0]
        self.p_stable_base_hip_rl = self.chain_base_hip_rl.fkin(self.qstable[self.rl[0]: self.rl[1]])[0]
        self.p_stable_base_hip_rr = self.chain_base_hip_rr.fkin(self.qstable[self.rr[0]: self.rr[1]])[0]

        self.p_stable_base_thigh_fl = self.chain_base_thigh_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]])[0]
        self.p_stable_base_thigh_fr = self.chain_base_thigh_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]])[0]
        self.p_stable_base_thigh_rl = self.chain_base_thigh_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]])[0]
        self.p_stable_base_thigh_rr = self.chain_base_thigh_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]])[0]

        self.p_stable_foot_fl = self.p_stable_base_foot_fl - self.p_stable_base_hip_fl
        self.p_stable_foot_fr = self.p_stable_base_foot_fr - self.p_stable_base_hip_fr
        self.p_stable_foot_rl = self.p_stable_base_foot_rl - self.p_stable_base_hip_rl
        self.p_stable_foot_rr = self.p_stable_base_foot_rr - self.p_stable_base_hip_rr

        self.p_stable_thigh_fl = self.p_stable_base_thigh_fl - self.p_stable_base_hip_fl
        self.p_stable_thigh_fr = self.p_stable_base_thigh_fr - self.p_stable_base_hip_fr
        self.p_stable_thigh_rl = self.p_stable_base_thigh_rl - self.p_stable_base_hip_rl
        self.p_stable_thigh_rr = self.p_stable_base_thigh_rr - self.p_stable_base_hip_rr

        self.p_stable = np.concatenate((self.p_stable_foot_fl, self.p_stable_foot_fr, self.p_stable_foot_rl, self.p_stable_foot_rr))
        self.pdlast = self.p_stable
        self.pd_leg_start = self.pdlast
        
        return self.p_stable

    def stand(self):
        
        q_joints_des = self.q0_joint
        return q_joints_des
    
    def walk_forward(self, t, dt):
        
        # Note: every cycle starts off with the front left forward

        def double_stance1(t_curr, T, start, end):
            
            x_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.pd_leg_start + np.array([-x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,])

            return pd_leg

        def stanceleft_swingright(t_curr, T, start, end):

            # calculate x_swing
            x_swing = start + (end - start) * (t_curr/T)
            

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[5], 0.1)

            # need to normalize x to be between 0 and 1
            x_curr = (x_swing - start)/(end - start)

            z_swing = bez.create_bezier(x_curr)

            # now we adjust pd_leg based on these values
            pd_leg = self.stance1 + np.array([-x_stance, 0.0, 0.0,
                                x_swing, 0.0, z_swing,
                                x_swing, 0.0, z_swing,
                                -x_stance, 0.0, 0.0])
            

            if self.cycle == 2:
                
                self.x_list.append(x_swing)
                self.z_list.append(z_swing)
            
            return pd_leg

        def double_stance2(t_curr, T, start, end):
            
            x_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.double2_start + np.array([-x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,
                                                   -x_curr, 0.0, 0.0,])

            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end):

            # even though the 8th order gives the nicest foot trajectory, the least error occurs with the
            # 4th order curve

            # calculate x_swing
            x_swing = start + (end - start) * (t_curr/T)

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[2], 0.1)

            # need to normalize x to be between 0 and 1
            x_curr = (x_swing - start)/(end - start)

            z_swing = bez.create_bezier(x_curr)
            # print(f'z_swing is: {z_swing}')

            # now we adjust pd_leg based on these values
            pd_leg = self.stance2 + np.array([x_swing, 0.0, z_swing,
                                              -x_stance, 0.0, 0.0,
                                              -x_stance, 0.0, 0.0,
                                              x_swing, 0.0, z_swing])
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            self.pd_leg_start = self.double1_start
            # self.pd_leg_start[2::3] = self.p_stable[2::3]
            self.cycle += 1

            if self.cycle == 0:
                print(f'Iteration {self.cycle}: current base position: {self.data.qpos[self.mj_idx.q_base_pos_idx]}')

            if self.cycle % 8 == 0:
                print(f'Iteration {self.cycle}: current base position: {self.data.qpos[self.mj_idx.q_base_pos_idx]}')

        t = t % self.cycle_len

        if t < self.cycle_len/8: # this lasts for 1/8th of the period

            # checking this method - this method seems to be the best but i don't know why :/
            if self.cycle == 0:
                pd_leg = self.pdlast
                self.stance1 = pd_leg
            else:
                pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
                self.stance1 = pd_leg

        # single stance for the left leg and swing for the right leg
        elif t < (4/8) * self.cycle_len: # this lasts 3/8th of the time
            # print('flopped here')
            pd_leg = stanceleft_swingright(t - self.double_stance_time, self.single_stance_time, 0, self.deltx)
            self.double2_start = pd_leg

        elif t < (5/8) * self.cycle_len:
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg

        elif t < self.cycle_len:
            pd_leg = swingleft_stanceright(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.deltx)
            self.double1_start = pd_leg

        else:
            print('discrete timing sucks')


        v_base_y_curr = self.data.qvel[self.mj_idx.VEL_Y]
        # print(f'current y-velocity: {v_base_y_curr}')
        print(f'current y-position: {self.data.qvel[self.mj_idx.POS_Y]}')
        v_base_y_desired = 0.0
        # kv = -0.009 # i think this is the best one so far lol
        kv = -0.05 # nvm this is the best one

        pd_leg[1] = pd_leg[1] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[4] = pd_leg[4] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[7] = pd_leg[7] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[10] = pd_leg[10] - kv*(v_base_y_curr - v_base_y_desired)


        theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        if None in theta_FL or None in theta_FR or None in theta_RL or None in theta_RR:
            print(f'pause here. failed at t = {t} \n {theta_FL}, {theta_FR}, {theta_RL}, {theta_RR}')


        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr
    
    def walk_reverse(self, t, dt):

        # Note: every cycle starts off with the front left forward

        def double_stance1(t_curr, T, start, end):
            
            x_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.pd_leg_start + np.array([x_curr, 0.0, 0.0,
                                                   x_curr, 0.0, 0.0,
                                                   x_curr, 0.0, 0.0,
                                                   x_curr, 0.0, 0.0,])

            return pd_leg

        def stanceleft_swingright(t_curr, T, start, end):

            # calculate x_swing
            x_swing = start + (end - start) * (t_curr/T)
            

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[5], 0.1)

            # need to normalize x to be between 0 and 1
            x_curr = (x_swing - start)/(end - start)

            z_swing = bez.create_bezier(x_curr)

            # now we adjust pd_leg based on these values
            pd_leg = self.stance1 + np.array([x_stance, 0.0, 0.0,
                                              -x_swing, 0.0, z_swing,
                                              -x_swing, 0.0, z_swing,
                                              x_stance, 0.0, 0.0])
            

            if self.cycle == 2:
                
                self.x_list.append(x_swing)
                self.z_list.append(z_swing)
            
            return pd_leg

        def double_stance2(t_curr, T, start, end):
            
            x_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.double2_start + np.array([x_curr, 0.0, 0.0,
                                                    x_curr, 0.0, 0.0,
                                                    x_curr, 0.0, 0.0,
                                                    x_curr, 0.0, 0.0,])

            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end):

            # even though the 8th order gives the nicest foot trajectory, the least error occurs with the
            # 4th order curve

            # calculate x_swing
            x_swing = start + (end - start) * (t_curr/T)

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[2], 0.1)

            # need to normalize x to be between 0 and 1
            x_curr = (x_swing - start)/(end - start)

            z_swing = bez.create_bezier(x_curr)
            # print(f'z_swing is: {z_swing}')

            # now we adjust pd_leg based on these values
            pd_leg = self.stance2 + np.array([-x_swing, 0.0, z_swing,
                                              x_stance, 0.0, 0.0,
                                              x_stance, 0.0, 0.0,
                                              -x_swing, 0.0, z_swing])
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            # print(f'checking x-positions: {self.pdlast[0]}, {self.pdlast[3]}, {self.pdlast[6]}, {self.pdlast[9]}')
            self.pd_leg_start = self.double1_start
            self.pd_leg_start[2::3] = self.p_stable[2::3]
            self.cycle += 1

            # if self.cycle % 8 == 0:
            #     print('stop here')

            # if self.cycle == 3:
            #     print(f'x: {self.x_list}')
            #     print(f'z: {self.z_list}')

        t = t % self.cycle_len

        if t < self.cycle_len/8: # this lasts for 1/8th of the period

            # # lazy fix
            # if self.cycle == 0:
            #     pd_leg = self.pdlast
            #     self.stance1 = pd_leg
            # else:
            #     pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
            #     self.stance1 = pd_leg

            # # smarter fix
            # pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
            # self.stance1 = pd_leg

            # checking this method - this method seems to be the best but i don't know why :/
            if self.cycle == 0:
                pd_leg = self.pdlast
                self.stance1 = pd_leg
            else:
                pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
                self.stance1 = pd_leg

        # single stance for the left leg and swing for the right leg
        elif t < (4/8) * self.cycle_len: # this lasts 3/8th of the time
            # print('flopped here')
            pd_leg = stanceleft_swingright(t - self.double_stance_time, self.single_stance_time, 0, self.deltx)
            self.double2_start = pd_leg

        elif t < (5/8) * self.cycle_len:
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg

        elif t < self.cycle_len:
            pd_leg = swingleft_stanceright(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.deltx)
            self.double1_start = pd_leg

        else:
            print('discrete timing sucks')

        v_base_y_curr = self.data.qvel[self.mj_idx.VEL_Y]
        # print(f'current y-velocity: {v_base_y_curr}')
        print(f'current y-position: {self.data.qpos[self.mj_idx.POS_Y]}')
        v_base_y_desired = 0.0
        # kv = -0.009 # i think this is the best one so far lol
        kv = -0.05 # nvm this is the best one

        pd_leg[1] = pd_leg[1] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[4] = pd_leg[4] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[7] = pd_leg[7] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[10] = pd_leg[10] - kv*(v_base_y_curr - v_base_y_desired)

        theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        if None in theta_FL or None in theta_FR or None in theta_RL or None in theta_RR:
            print(f'pause here. failed at t = {t} \n {theta_FL}, {theta_FR}, {theta_RL}, {theta_RR}')


        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr
        
    def walk_side_step_left(self, t, dt):

        # Note: every cycle starts off with the front left forward

        def double_stance1(t_curr, T, start, end):
            
            y_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward


            pd_leg = self.pd_leg_start + np.array([0.0, -y_curr, 0.0,
                                                   0.0, -y_curr, 0.0,
                                                   0.0, -y_curr, 0.0,
                                                   0.0, -y_curr, 0.0,])

            return pd_leg

        def stanceleft_swingright(t_curr, T, start, end):

            # calculate x_swing
            y_swing = start + (end - start) * (t_curr/T)
            

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[5], 0.1)

            # need to normalize x to be between 0 and 1
            y_curr = (y_swing - start)/(end - start)

            z_swing = bez.create_bezier(y_curr)

            pd_leg = self.stance1 + np.array([0.0, -y_stance, 0.0,
                                              0.0, y_swing, z_swing,
                                              0.0, y_swing, z_swing,
                                              0.0, -y_stance, 0.0])
            
            return pd_leg

        def double_stance2(t_curr, T, start, end):
            
            y_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.double2_start + np.array([0.0, -y_curr, 0.0,
                                                    0.0, -y_curr, 0.0,
                                                    0.0, -y_curr, 0.0,
                                                    0.0, -y_curr, 0.0,])

            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end):

            # even though the 8th order gives the nicest foot trajectory, the least error occurs with the
            # 4th order curve

            # calculate x_swing
            y_swing = start + (end - start) * (t_curr/T)

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[2], 0.1)

            # need to normalize x to be between 0 and 1
            y_curr = (y_swing - start)/(end - start)

            z_swing = bez.create_bezier(y_curr)
        

            pd_leg = self.stance2 + np.array([0.0, y_swing, z_swing,
                                              0.0, -y_stance, 0.0,
                                              0.0, -y_stance, 0.0,
                                              0.0, y_swing, z_swing])
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            # print(f'checking x-positions: {self.pdlast[0]}, {self.pdlast[3]}, {self.pdlast[6]}, {self.pdlast[9]}')
            self.pd_leg_start = self.double1_start
            # self.pd_leg_start[2::3] = self.p_stable[2::3]
            self.cycle += 1


        t = t % self.cycle_len

        if t < self.double_stance_time: 

            # smarter fix
            pd_leg = double_stance1(t, self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            self.stance1 = pd_leg

            # # checking this method - this method seems to be the best but i don't know why :/
            # if self.cycle == 0:
            #     pd_leg = self.pdlast
            #     self.stance1 = pd_leg
            # else:
            #     pd_leg = double_stance1(t, self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            #     self.stance1 = pd_leg

        # single stance for the left leg and swing for the right leg
        elif t < (self.double_stance_time + self.single_stance_time): # this lasts 3/8th of the time
            # print('flopped here')
            pd_leg = stanceleft_swingright(t - self.double_stance_time, self.single_stance_time, 0, self.delty)
            self.double2_start = pd_leg

        elif t < (self.double_stance_time + self.double_stance_time + self.single_stance_time):
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg

        elif t < self.cycle_len:
            pd_leg = swingleft_stanceright(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.delty)
            self.double1_start = pd_leg

        else:
            print('discrete timing sucks')

        v_base_x_curr = self.data.qvel[self.mj_idx.VEL_X]
        # print(f'current y-velocity: {v_base_y_curr}')
        print(f'current x-position: {self.data.qpos[self.mj_idx.POS_X]}')
        v_base_x_desired = 0.0
        # kv = -0.009 # i think this is the best one so far lol
        kv = -0.0085 # nvm this is the best one

        pd_leg[0] = pd_leg[0] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[3] = pd_leg[3] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[6] = pd_leg[6] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[9] = pd_leg[9] - kv*(v_base_x_curr - v_base_x_desired)


        theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        if None in theta_FL or None in theta_FR or None in theta_RL or None in theta_RR:
            print(f'pause here. failed at t = {t} \n {theta_FL}, {theta_FR}, {theta_RL}, {theta_RR}')


        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr 
    
    def walk_side_step_right(self, t, dt):
        # Note: every cycle starts off with the front left forward

        def double_stance1(t_curr, T, start, end):
            
            y_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward


            pd_leg = self.pd_leg_start + np.array([0.0, y_curr, 0.0,
                                                   0.0, y_curr, 0.0,
                                                   0.0, y_curr, 0.0,
                                                   0.0, y_curr, 0.0,])

            return pd_leg

        def stanceleft_swingright(t_curr, T, start, end):

            # calculate x_swing
            y_swing = start + (end - start) * (t_curr/T)
            

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve
            # z_swing = -x_swing * (x_swing - end) * 30

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[5], 0.1)

            # need to normalize x to be between 0 and 1
            y_curr = (y_swing - start)/(end - start)

            z_swing = bez.create_bezier(y_curr)

            pd_leg = self.stance1 + np.array([0.0, y_stance, 0.0,
                                              0.0, -y_swing, z_swing,
                                              0.0, -y_swing, z_swing,
                                              0.0, y_stance, 0.0])
            
            return pd_leg

        def double_stance2(t_curr, T, start, end):
            
            y_curr = start + (end - start) * (t_curr/T)

            # this is applied to both legs, both legs move backward

            pd_leg = self.double2_start + np.array([0.0, y_curr, 0.0,
                                                    0.0, y_curr, 0.0,
                                                    0.0, y_curr, 0.0,
                                                    0.0, y_curr, 0.0,])

            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end):

            # even though the 8th order gives the nicest foot trajectory, the least error occurs with the
            # 4th order curve

            # calculate x_swing
            y_swing = start + (end - start) * (t_curr/T)

            # calculate x_stance
            end_stance = end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (end_stance - start) * (t_curr/T)

            # calculate z_swing using bezier curve

            # chose and arbitrary height for step height
            bez = Bezier2D(start, end, self.p_stable[2], 0.1)

            # need to normalize x to be between 0 and 1
            y_curr = (y_swing - start)/(end - start)

            z_swing = bez.create_bezier(y_curr)
        

            pd_leg = self.stance2 + np.array([0.0, -y_swing, z_swing,
                                              0.0, y_stance, 0.0,
                                              0.0, y_stance, 0.0,
                                              0.0, -y_swing, z_swing])
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            # print(f'checking x-positions: {self.pdlast[0]}, {self.pdlast[3]}, {self.pdlast[6]}, {self.pdlast[9]}')
            self.pd_leg_start = self.double1_start
            # self.pd_leg_start[2::3] = self.p_stable[2::3]
            self.cycle += 1


        t = t % self.cycle_len

        if t < self.double_stance_time: 

            # smarter fix
            pd_leg = double_stance1(t, self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            self.stance1 = pd_leg

            # # checking this method - this method seems to be the best but i don't know why :/
            # if self.cycle == 0:
            #     pd_leg = self.pdlast
            #     self.stance1 = pd_leg
            # else:
            #     pd_leg = double_stance1(t, self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            #     self.stance1 = pd_leg

        # single stance for the left leg and swing for the right leg
        elif t < (self.double_stance_time + self.single_stance_time): # this lasts 3/8th of the time
            # print('flopped here')
            pd_leg = stanceleft_swingright(t - self.double_stance_time, self.single_stance_time, 0, self.delty)
            self.double2_start = pd_leg

        elif t < (self.double_stance_time + self.double_stance_time + self.single_stance_time):
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.delty * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg

        elif t < self.cycle_len:
            pd_leg = swingleft_stanceright(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.delty)
            self.double1_start = pd_leg

        else:
            print('discrete timing sucks')

        v_base_x_curr = self.data.qvel[self.mj_idx.VEL_X]
        # print(f'current y-velocity: {v_base_y_curr}')
        print(f'current x-position: {self.data.qpos[self.mj_idx.POS_X]}')
        v_base_x_desired = 0.0
        # # kv = -0.009 # i think this is the best one so far lol
        kv = -0.0002 # nvm this is the best one

        pd_leg[0] = pd_leg[0] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[3] = pd_leg[3] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[6] = pd_leg[6] - kv*(v_base_x_curr - v_base_x_desired)
        pd_leg[9] = pd_leg[9] - kv*(v_base_x_curr - v_base_x_desired)


        theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        if None in theta_FL or None in theta_FR or None in theta_RL or None in theta_RR:
            print(f'pause here. failed at t = {t} \n {theta_FL}, {theta_FR}, {theta_RL}, {theta_RR}')


        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr
    
    def turn_clockwise(self, t, dt):
        def double_stance1(t_curr, T, start, end):

            x_start = 0
            x_end = self.r * cos(end) - self.r
            x_stance = x_start + (x_end - x_start) * (t_curr/T)

            y_start = 0
            y_end = self.r * sin(end) 
            y_stance = y_start + (y_end - y_start) * (t_curr/T)

            pd_leg = self.pd_leg_start + np.array([x_stance, y_stance, 0.0,
                                                   x_stance, y_stance, 0.0,
                                                -x_stance,  -y_stance, 0.0,
                                                    -x_stance,  -y_stance, 0.0,])
            
            return pd_leg


        def swingleft_stanceright(t_curr, T, start, end):


            # math for the swing legs
            x_swing_start = 0
            x_swing_end = self.r * cos(end) - self.r
            x_swing = x_swing_start + (x_swing_end - x_swing_start) * (t_curr/T)

            y_swing_start = 0
            y_swing_end = self.r * sin(end)
            y_swing = y_swing_start + (y_swing_end - y_swing_start) * (t_curr/T)
            
            # normalize x_swing for calculating the bezier curve
            x_swing_norm = (x_swing - x_swing_start)/(x_swing_end - x_swing_start)
            y_swing_norm = (y_swing - y_swing_start)/(y_swing_end - y_swing_start)

            bez = Bezier3D(0.0, x_swing_end, self.p_stable[2], 0.1, 0.0, y_swing_end)
            z_swing = bez.create_bezier(x_swing_norm)[-1]

            # math for the stance legs
            x_end_stance = x_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (x_end_stance - start) * (t_curr/T)

            y_end_stance = y_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (y_end_stance - start) * (t_curr/T)

            # adjust pd_leg based on these values

            # think of it like this. FL goes up and in, so the FR has to go out down and out to balance

            pd_leg = self.stance1 + np.array([-x_swing,  -y_swing, z_swing,
                                               x_stance,  y_stance, 0.0,
                                              -x_stance, -y_stance, 0.0,
                                               x_swing,   y_swing, z_swing])

            # print(f'[x_swing, y_swing, z_swing]: {[x_swing, y_swing, z_swing]}')
            
            return pd_leg
        
        def double_stance2(t_curr, T, start, end):

            x_start = 0
            x_end = self.r * cos(end) - self.r
            x_stance = x_start + (x_end - x_start) * (t_curr/T)

            y_start = 0
            y_end = self.r * sin(end) 
            y_stance = y_start + (y_end - y_start) * (t_curr/T)

            pd_leg = self.double2_start + np.array([  x_stance,  y_stance, 0.0,
                                                      x_stance,  y_stance, 0.0,
                                                     -x_stance, -y_stance, 0.0,
                                                     -x_stance, -y_stance, 0.0,])
            
            return pd_leg
        
        def swingright_stanceleft(t_curr, T, start, end):

            # math for the swing legs
            x_swing_start = 0
            x_swing_end = self.r * cos(end) - self.r
            x_swing = x_swing_start + (x_swing_end - x_swing_start) * (t_curr/T)

            y_swing_start = 0
            y_swing_end = self.r * sin(end)
            y_swing = y_swing_start + (y_swing_end - y_swing_start) * (t_curr/T)
            
            # normalize x_swing for calculating the bezier curve
            x_swing_norm = (x_swing - x_swing_start)/(x_swing_end - x_swing_start)
            y_swing_norm = (y_swing - y_swing_start)/(y_swing_end - y_swing_start)

            bez = Bezier3D(0.0, x_swing_end, self.p_stable[2], 0.1, 0.0, y_swing_end)
            z_swing = bez.create_bezier(x_swing_norm)[-1]

            # math for the stance legs
            x_end_stance = x_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (x_end_stance - start) * (t_curr/T)

            y_end_stance = y_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (y_end_stance - start) * (t_curr/T)

            # adjust pd_leg based on these values
            pd_leg = self.stance2 + np.array([x_stance,  y_stance, 0.0,
                                             -x_swing,  -y_swing, z_swing,
                                              x_swing,   y_swing, z_swing,
                                             -x_stance, -y_stance, 0.0])

            print(f'[x_swing, y_swing, z_swing]: {[x_swing, y_swing, z_swing]}')
            
            return pd_leg

        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            self.cycle += 1
            self.pd_leg_start = self.double1_start


        t = t % self.cycle_len

        if t < self.double_stance_time:

            # in double stance mode

            if self.cycle == 0:
                pd_leg = self.pdlast
                self.stance1 = pd_leg

            else:
                pd_leg = double_stance1(t, self.double_stance_time, 0, self.delta_theta * (self.double_stance_time/self.cycle_len))
                self.stance1 = pd_leg
        
        elif t < (self.double_stance_time + self.single_stance_time):

            # swing the left leg, stance the right leg
            pd_leg = swingleft_stanceright(t - self.double_stance_time, self.single_stance_time, 0, self.delta_theta)
            self.double2_start = pd_leg

        elif t < (2*self.double_stance_time + self.single_stance_time):

            # both stance, re-adjust th ebody position
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.delta_theta * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg
        
        elif t < self.cycle_len:
            pd_leg = swingright_stanceleft(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.delta_theta)
            self.double1_start = pd_leg

        else:
            pd_leg = self.pdlast


        theta_FL = NewtonRaphson(self.p_stable[0:3], pd_leg[0:3], self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.p_stable[3:6], pd_leg[3:6], self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.p_stable[6:9], pd_leg[6:9], self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.p_stable[9:12], pd_leg[9:12], self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr
        
    def turn_counterclockwise(self, t, dt):

        def double_stance1(t_curr, T, start, end):

            x_start = 0
            x_end = self.r * cos(end) - self.r
            x_stance = x_start + (x_end - x_start) * (t_curr/T)

            y_start = 0
            y_end = self.r * sin(end) 
            y_stance = y_start + (y_end - y_start) * (t_curr/T)

            pd_leg = self.pd_leg_start + np.array([-x_stance, -y_stance, 0.0,
                                                   -x_stance, -y_stance, 0.0,
                                                    x_stance,  y_stance, 0.0,
                                                    x_stance,  y_stance, 0.0,])
            
            return pd_leg


        def swingleft_stanceright(t_curr, T, start, end):


            # math for the swing legs
            x_swing_start = 0
            x_swing_end = self.r * cos(end) - self.r
            x_swing = x_swing_start + (x_swing_end - x_swing_start) * (t_curr/T)

            y_swing_start = 0
            y_swing_end = self.r * sin(end)
            y_swing = y_swing_start + (y_swing_end - y_swing_start) * (t_curr/T)
            
            # normalize x_swing for calculating the bezier curve
            x_swing_norm = (x_swing - x_swing_start)/(x_swing_end - x_swing_start)
            y_swing_norm = (y_swing - y_swing_start)/(y_swing_end - y_swing_start)

            bez = Bezier3D(0.0, x_swing_end, self.p_stable[2], 0.1, 0.0, y_swing_end)
            z_swing = bez.create_bezier(x_swing_norm)[-1]

            # math for the stance legs
            x_end_stance = x_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (x_end_stance - start) * (t_curr/T)

            y_end_stance = y_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (y_end_stance - start) * (t_curr/T)

            # adjust pd_leg based on these values

            # think of it like this. FL goes up and in, so the FR has to go out down and out to balance

            pd_leg = self.stance1 + np.array([x_swing, y_swing, z_swing,
                                             -x_stance, -y_stance, 0.0,
                                              x_stance, y_stance, 0.0,
                                             -x_swing, -y_swing, z_swing])

            # print(f'[x_swing, y_swing, z_swing]: {[x_swing, y_swing, z_swing]}')
            
            return pd_leg
        
        def double_stance2(t_curr, T, start, end):

            x_start = 0
            x_end = self.r * cos(end) - self.r
            x_stance = x_start + (x_end - x_start) * (t_curr/T)

            y_start = 0
            y_end = self.r * sin(end) 
            y_stance = y_start + (y_end - y_start) * (t_curr/T)

            pd_leg = self.double2_start + np.array([-x_stance, -y_stance, 0.0,
                                                    -x_stance,  -y_stance, 0.0,
                                                     x_stance,  y_stance, 0.0,
                                                     x_stance,  y_stance, 0.0,])
            
            return pd_leg
        
        def swingright_stanceleft(t_curr, T, start, end):

            # math for the swing legs
            x_swing_start = 0
            x_swing_end = self.r * cos(end) - self.r
            x_swing = x_swing_start + (x_swing_end - x_swing_start) * (t_curr/T)

            y_swing_start = 0
            y_swing_end = self.r * sin(end)
            y_swing = y_swing_start + (y_swing_end - y_swing_start) * (t_curr/T)
            
            # normalize x_swing for calculating the bezier curve
            x_swing_norm = (x_swing - x_swing_start)/(x_swing_end - x_swing_start)
            y_swing_norm = (y_swing - y_swing_start)/(y_swing_end - y_swing_start)

            bez = Bezier3D(0.0, x_swing_end, self.p_stable[2], 0.1, 0.0, y_swing_end)
            z_swing = bez.create_bezier(x_swing_norm)[-1]

            # math for the stance legs
            x_end_stance = x_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            x_stance = start + (x_end_stance - start) * (t_curr/T)

            y_end_stance = y_swing_end * (1 - (2 * self.double_stance_time)/self.cycle_len)
            y_stance = start + (y_end_stance - start) * (t_curr/T)

            # adjust pd_leg based on these values
            pd_leg = self.stance2 + np.array([-x_stance, -y_stance, 0.0,
                                               x_swing, y_swing, z_swing,
                                              -x_swing, -y_swing, z_swing,
                                               x_stance, y_stance, 0.0])

            print(f'[x_swing, y_swing, z_swing]: {[x_swing, y_swing, z_swing]}')
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            self.cycle += 1
            self.pd_leg_start = self.double1_start


        t = t % self.cycle_len

        if t < self.double_stance_time:

            # in double stance mode

            if self.cycle == 0:
                pd_leg = self.pdlast
                self.stance1 = pd_leg

            else:
                pd_leg = double_stance1(t, self.double_stance_time, 0, self.delta_theta * (self.double_stance_time/self.cycle_len))
                self.stance1 = pd_leg
        
        elif t < (self.double_stance_time + self.single_stance_time):

            # swing the left leg, stance the right leg
            pd_leg = swingleft_stanceright(t - self.double_stance_time, self.single_stance_time, 0, self.delta_theta)
            self.double2_start = pd_leg

        elif t < (2*self.double_stance_time + self.single_stance_time):

            # both stance, re-adjust th ebody position
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, self.delta_theta * (self.double_stance_time/self.cycle_len))
            self.stance2 = pd_leg
        
        elif t < self.cycle_len:
            pd_leg = swingright_stanceleft(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, self.delta_theta)
            self.double1_start = pd_leg

        else:
            pd_leg = self.pdlast


        theta_FL = NewtonRaphson(self.p_stable[0:3], pd_leg[0:3], self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.p_stable[3:6], pd_leg[3:6], self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.p_stable[6:9], pd_leg[6:9], self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.p_stable[9:12], pd_leg[9:12], self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg
        
        return self.qcurr
    
    def reset(self, t_curr, T):

        # get the current leg position using self.pdlast
        p0_FL_bf = self.chain_base_foot_fl.fkin(self.qcurr[self.fl[0]: self.fl[-1]+1])[0]
        p0_FR_bf = self.chain_base_foot_fr.fkin(self.qcurr[self.fr[0]: self.fr[-1]+1])[0]
        p0_RL_bf = self.chain_base_foot_rl.fkin(self.qcurr[self.rl[0]: self.rl[-1]+1])[0]
        p0_RR_bf = self.chain_base_foot_rr.fkin(self.qcurr[self.rr[0]: self.rr[-1]+1])[0]

        p0_FL_bh = self.chain_base_hip_fl.fkin(self.qcurr[self.fl[0]:self.fl[1]])[0]
        p0_FR_bh = self.chain_base_hip_fr.fkin(self.qcurr[self.fr[0]:self.fr[1]])[0]
        p0_RL_bh = self.chain_base_hip_rl.fkin(self.qcurr[self.rl[0]:self.rl[1]])[0]
        p0_RR_bh = self.chain_base_hip_rr.fkin(self.qcurr[self.rr[0]:self.rr[1]])[0]

        p0_FL_bt = self.chain_base_thigh_fl.fkin(self.qcurr[self.fl[0]: self.fl[-1]])[0]
        p0_FR_bt = self.chain_base_thigh_fr.fkin(self.qcurr[self.fr[0]: self.fr[-1]])[0]
        p0_RL_bt = self.chain_base_thigh_rl.fkin(self.qcurr[self.rl[0]: self.rl[-1]])[0]
        p0_RR_bt = self.chain_base_thigh_rr.fkin(self.qcurr[self.rr[0]: self.rr[-1]])[0]

        # get the current foot positions
        p0_FLx = p0_FL_bf[0] - p0_FL_bh[0]
        p0_FRx = p0_FR_bf[0] - p0_FR_bh[0]
        p0_RLx = p0_RL_bf[0] - p0_RL_bh[0]
        p0_RRx = p0_RR_bf[0] - p0_RR_bh[0]

        p0_FLy = p0_FL_bf[1] - p0_FL_bh[1]
        p0_FRy = p0_FR_bf[1] - p0_FR_bh[1]
        p0_RLy = p0_RL_bf[1] - p0_RL_bh[1]
        p0_RRy = p0_RR_bf[1] - p0_RR_bh[1]

        p0_FLz = p0_FL_bf[2] - p0_FL_bh[2]
        p0_FRz = p0_FR_bf[2] - p0_FR_bh[2]
        p0_RLz = p0_RL_bf[2] - p0_RL_bh[2]
        p0_RRz = p0_RR_bf[2] - p0_RR_bh[2]

        # get the current thigh positions (aka where we wanna go)
        pf_FLx = p0_FL_bt[0] - p0_FL_bh[0]
        pf_FRx = p0_FR_bt[0] - p0_FR_bh[0]
        pf_RLx = p0_RL_bt[0] - p0_RL_bh[0]
        pf_RRx = p0_RR_bt[0] - p0_RR_bh[0]

        pf_FLy = p0_FL_bt[1] - p0_FL_bh[1]
        pf_FRy = p0_FR_bt[1] - p0_FR_bh[1]
        pf_RLy = p0_RL_bt[1] - p0_RL_bh[1]
        pf_RRy = p0_RR_bt[1] - p0_RR_bh[1]

        pf_FLz = p0_FL_bt[2] - p0_FL_bh[2]
        pf_FRz = p0_FR_bt[2] - p0_FR_bh[2]
        pf_RLz = p0_RL_bt[2] - p0_RL_bh[2]
        pf_RRz = p0_RR_bt[2] - p0_RR_bh[2]

        
        self.p0_foot_fl = [pf_FLx, pf_FLy, self.pdlast[2]]
        self.p0_foot_fr = [pf_FRx, pf_FRy, self.pdlast[5]]
        self.p0_foot_rl = [pf_RLx, pf_RLy, self.pdlast[8]]
        self.p0_foot_rr = [pf_RRx, pf_RRy, self.pdlast[11]]



        alpha = t_curr/T

        pdes_x_FL = (1 - alpha) * p0_FLx + alpha * pf_FLx
        pdes_y_FL = (1 - alpha) * p0_FLy + alpha * pf_FLy
        pcurr_FL = [pdes_x_FL, pdes_y_FL, self.pdlast[2]]

        pdes_x_FR = (1 - alpha) * p0_FRx + alpha * pf_FRx
        pdes_y_FR = (1 - alpha) * p0_FRy + alpha * pf_FRy
        pcurr_FR = [pdes_x_FR, pdes_y_FR, self.pdlast[5]]

        pdes_x_RL = (1 - alpha) * p0_RLx + alpha * pf_RLx
        pdes_y_RL = (1 - alpha) * p0_RLy + alpha * pf_RLy
        pcurr_RL = [pdes_x_RL, pdes_y_RL, self.pdlast[8]]

        pdes_x_RR = (1 - alpha) * p0_RRx + alpha * pf_RRx
        pdes_y_RR = (1 - alpha) * p0_RRy + alpha * pf_RRy
        pcurr_RR = [pdes_x_RR, pdes_y_RR, self.pdlast[11]]

        # calculate the inverse kinematics
        theta_FL = NewtonRaphson(p0_FL_bf, pcurr_FL, self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(p0_FR_bf, pcurr_FR, self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(p0_RL_bf, pcurr_RL, self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(p0_RR_bf, pcurr_RR, self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        self.qreset = [theta_FL[0], theta_FL[1], theta_FL[2],
                       theta_FR[0], theta_FR[1], theta_FR[2],
                       theta_RL[0], theta_RL[1], theta_RL[2],
                       theta_RR[0], theta_RR[1], theta_RR[2]]
        
        return self.qreset

if __name__=="__main__":

    bot = QuadrupedController('./config/go2_config.yaml')
    joints = bot.stand()
    print(joints)