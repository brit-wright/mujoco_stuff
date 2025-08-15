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

# SEMI_MIN = 0.00194
# SEMI_MAJ = 0.0054

# class QuadrupedController:

#     def __init__(self, data, mj_idx, q0_joint):

#         self.data = data
#         self.mj_idx = mj_idx

#         # initialize the joint indices
#         self.fl = [0, 1, 2]
#         self.fr = [3, 4, 5]
#         self.rl = [6, 7, 8]
#         self.rr = [9, 10, 11]

#         self.x_list = []
#         self.t_list = []
#         self.z_list = []
        
#         # define the initial joint positions
#         q = data.qpos
#         self.q_joints = q[mj_idx.q_joint_idx]
#         self.qstable = self.q_joints
#         self.qcurr = self.qstable

#         self.q0_joint = q0_joint

#         self.cycle = 0
#         self.cycle_forward = 0

#         self.urdf = './models/go2/go2.urdf'

#         self.cycle_len = 0.4

#         # set up the kinematic chains
#         # chains from base to foot
#         self.chain_base_foot_fl = KinematicChain('base', 'FL_foot', self.jointnames()[0:3], self.urdf)
#         self.chain_base_foot_fr = KinematicChain('base', 'FR_foot', self.jointnames()[3:6], self.urdf)
#         self.chain_base_foot_rl = KinematicChain('base', 'RL_foot', self.jointnames()[6:9], self.urdf)
#         self.chain_base_foot_rr = KinematicChain('base', 'RR_foot', self.jointnames()[9:12], self.urdf)


#         # chains from base to hip
#         self.chain_base_hip_fl = KinematicChain('base', 'FL_hip', self.jointnames()[0:1], self.urdf)
#         self.chain_base_hip_fr = KinematicChain('base', 'FR_hip', self.jointnames()[3:4], self.urdf)
#         self.chain_base_hip_rl = KinematicChain('base', 'RL_hip', self.jointnames()[6:7], self.urdf)
#         self.chain_base_hip_rr = KinematicChain('base', 'RR_hip', self.jointnames()[9:10], self.urdf)


#         # chains from base to thigh
#         self.chain_base_thigh_fl = KinematicChain('base', 'FL_thigh', self.jointnames()[0:2], self.urdf)
#         self.chain_base_thigh_fr = KinematicChain('base', 'FR_thigh', self.jointnames()[3:5], self.urdf)
#         self.chain_base_thigh_rl = KinematicChain('base', 'RL_thigh', self.jointnames()[6:8], self.urdf)
#         self.chain_base_thigh_rr = KinematicChain('base', 'RR_thigh', self.jointnames()[9:11], self.urdf)

#         # get the initial position of the foot
#         self.p0_base_foot_fl = self.chain_base_foot_fl.fkin(self.q_joints[self.fl[0]: self.fl[-1]+1])[0]
#         self.p0_base_hip_fl = self.chain_base_hip_fl.fkin(self.q_joints[self.fl[0]:self.fl[1]])[0]
#         self.p0_base_thigh_fl = self.chain_base_thigh_fl.fkin(self.q_joints[self.fl[0]: self.fl[-1]])[0]

#         self.p0_foot_fl = self.p0_base_foot_fl - self.p0_base_hip_fl
#         self.p0_thigh_fl = self.p0_base_thigh_fl - self.p0_base_hip_fl


#         self.p0_base_foot_fr = self.chain_base_foot_fr.fkin(self.q_joints[self.fr[0]: self.fr[-1]+1])[0]
#         self.p0_base_hip_fr = self.chain_base_hip_fr.fkin(self.q_joints[self.fr[0]:self.fr[1]])[0]
#         self.p0_base_thigh_fr = self.chain_base_thigh_fr.fkin(self.q_joints[self.fr[0]: self.fr[-1]])[0]

#         self.p0_foot_fr = self.p0_base_foot_fr - self.p0_base_hip_fr
#         self.p0_thigh_fr = self.p0_base_thigh_fr - self.p0_base_hip_fr


#         self.p0_base_foot_rl = self.chain_base_foot_rl.fkin(self.q_joints[self.rl[0]: self.rl[-1]+1])[0]
#         self.p0_base_hip_rl = self.chain_base_hip_rl.fkin(self.q_joints[self.rl[0]:self.rl[1]])[0]
#         self.p0_base_thigh_rl = self.chain_base_thigh_rl.fkin(self.q_joints[self.rl[0]: self.rl[-1]])[0]

#         self.p0_foot_rl = self.p0_base_foot_rl - self.p0_base_hip_rl
#         self.p0_thigh_rl = self.p0_base_thigh_rl - self.p0_base_hip_rl


#         self.p0_base_foot_rr = self.chain_base_foot_rr.fkin(self.q_joints[self.rr[0]: self.rr[-1]+1])[0]
#         self.p0_base_hip_rr = self.chain_base_hip_rr.fkin(self.q_joints[self.rr[0]:self.rr[1]])[0]
#         self.p0_base_thigh_rr = self.chain_base_thigh_rr.fkin(self.q_joints[self.rr[0]: self.rr[-1]])[0]

#         self.p0_foot_rr = self.p0_base_foot_rr - self.p0_base_hip_rr
#         self.p0_thigh_rr = self.p0_base_thigh_rr - self.p0_base_hip_rr

#         self.l_thigh = 0.213
#         self.l_calf = 0.213

#         # Define the stance timings from self.cycle_len
#         self.single_stance_time = 3 * self.cycle_len/8
#         self.double_stance_time = self.cycle_len/8

#         self.rot_speed = 0.8
#         self.delta_theta = self.rot_speed * self.single_stance_time

#         self.r = sqrt((0.1934)**2 + (0.0955+0.0465)**2)

#         self.all_four_turn = False
#         self.all_four_walk = False
#         self.is_stable = False

#         self.zcomm = 0.35
#         self.mode = ''

#         self.T_stab = 0.35
#         self.T_reset = 0.35

#         self.recovery_mode = False

#         self.forward_reset_check = False

#     # jointnames helper function
#     def jointnames(self):
#         # 12 joints
#         return['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
#                 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
#                 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
#                 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    
#     def choose_gait_start(self):
#         self.p_stable_base_foot_fl = self.chain_base_foot_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]+1])[0]
#         self.p_stable_base_foot_fr = self.chain_base_foot_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]+1])[0]
#         self.p_stable_base_foot_rl = self.chain_base_foot_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]+1])[0]
#         self.p_stable_base_foot_rr = self.chain_base_foot_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]+1])[0]

#         self.p_stable_base_hip_fl = self.chain_base_hip_fl.fkin(self.qstable[self.fl[0]: self.fl[1]])[0]
#         self.p_stable_base_hip_fr = self.chain_base_hip_fr.fkin(self.qstable[self.fr[0]: self.fr[1]])[0]
#         self.p_stable_base_hip_rl = self.chain_base_hip_rl.fkin(self.qstable[self.rl[0]: self.rl[1]])[0]
#         self.p_stable_base_hip_rr = self.chain_base_hip_rr.fkin(self.qstable[self.rr[0]: self.rr[1]])[0]

#         self.p_stable_base_thigh_fl = self.chain_base_thigh_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]])[0]
#         self.p_stable_base_thigh_fr = self.chain_base_thigh_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]])[0]
#         self.p_stable_base_thigh_rl = self.chain_base_thigh_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]])[0]
#         self.p_stable_base_thigh_rr = self.chain_base_thigh_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]])[0]

#         self.p_stable_foot_fl = self.p_stable_base_foot_fl - self.p_stable_base_hip_fl
#         self.p_stable_foot_fr = self.p_stable_base_foot_fr - self.p_stable_base_hip_fr
#         self.p_stable_foot_rl = self.p_stable_base_foot_rl - self.p_stable_base_hip_rl
#         self.p_stable_foot_rr = self.p_stable_base_foot_rr - self.p_stable_base_hip_rr

#         self.p_stable_thigh_fl = self.p_stable_base_thigh_fl - self.p_stable_base_hip_fl
#         self.p_stable_thigh_fr = self.p_stable_base_thigh_fr - self.p_stable_base_hip_fr
#         self.p_stable_thigh_rl = self.p_stable_base_thigh_rl - self.p_stable_base_hip_rl
#         self.p_stable_thigh_rr = self.p_stable_base_thigh_rr - self.p_stable_base_hip_rr

#         self.p_stable = np.concatenate((self.p_stable_foot_fl, self.p_stable_foot_fr, self.p_stable_foot_rl, self.p_stable_foot_rr))
#         self.pdlast = self.p_stable
#         self.pd_leg_start = self.pdlast
        
#         return self.p_stable

#     def stand(self):
        
#         q_joints_des = self.q0_joint
#         return q_joints_des
    
#     def walk_and_turn(self, t, commands, yaw_curr):

#         # NOTE: yaw_curr is supposed to be used for walking and turning
        
#         # # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
#         # v_foot_fl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot[0:2]))[0:2]
#         # v_foot_fr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot[0:2]))[0:2]
#         # v_foot_rl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot[0:2]))[0:2]
#         # v_foot_rr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot[0:2]))[0:2]

#         # # from these velocities calculate the ending foot positions, gives x and y
#         # pos_foot_fl = v_foot_fl * self.single_stance_time
#         # pos_foot_fr = v_foot_fr * self.single_stance_time
#         # pos_foot_rl = v_foot_rl * self.single_stance_time
#         # pos_foot_rr = v_foot_rr * self.single_stance_time
        
#         def double_stance1(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):
            
#             print('double stance 1')

#             # start by calculating the x values as an interpolation between the start and end
#             x_stance_fl = start + (end_fl[0] - start) * (t_curr/T)
#             x_stance_fr = start + (end_fr[0] - start) * (t_curr/T)
#             x_stance_rl = start + (end_rl[0] - start) * (t_curr/T)
#             x_stance_rr = start + (end_rr[0] - start) * (t_curr/T)

#             # calculate the y values as an interpolation between the start and end
#             y_stance_fl = start + (end_fl[1] - start) * (t_curr/T)
#             y_stance_fr = start + (end_fr[1] - start) * (t_curr/T)
#             y_stance_rl = start + (end_rl[1] - start) * (t_curr/T)
#             y_stance_rr = start + (end_rr[1] - start) * (t_curr/T)

#             pd_leg = self.pd_leg_start + np.array([x_stance_fl, y_stance_fl, 0.0,
#                                                    x_stance_fr, y_stance_fr, 0.0,
#                                                    x_stance_rl, y_stance_rl, 0.0,
#                                                    x_stance_rr, y_stance_rr, 0.0,])
            
#             return pd_leg

#         def swingleft_stanceright(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):

#             print('swing left, stance right')
            
#             # start by calculating the positions for the swing legs - fl, rr
#             x_swing_fl = start + (end_fl[0] - start) * (t_curr/T)
#             x_swing_rr = start + (end_fl[0] - start) * (t_curr/T)

#             y_swing_fl = start + (end_fl[1] - start) * (t_curr/T)
#             y_swing_rr = start + (end_fl[1] - start) * (t_curr/T)

#             # normalize the swing values to get the z-values from Bezier curves
#             x_swing_norm_fl = (x_swing_fl - 0)/(end_fl[0] - 0)
#             x_swing_norm_rr = (x_swing_fl - 0)/(end_fl[0] - 0)
            
#             bez_fl = Bezier3D(0.0, end_fl[0], self.p_stable[2], 0.1, 0.0, y_swing_fl)
#             z_swing_fl = bez_fl.create_bezier(x_swing_norm_fl)[-1]

#             bez_rr = Bezier3D(0.0, end_rr[0], self.p_stable[2], 0.1, 0.0, y_swing_rr)
#             z_swing_rr = bez_rr.create_bezier(x_swing_norm_rr)[-1]

#             # next calculate the positions for the stance legs - fr, rl
#             end_stance_fr = end_fr * (1 - (2 * self.double_stance_time)/self.cycle_len)
#             x_stance_fr = start + (end_stance_fr[0] - start) * (t_curr/T)
#             y_stance_fr = start + (end_stance_fr[1] - start) * (t_curr/T)

#             end_stance_rl = end_rl * (1 - (2 * self.double_stance_time)/self.cycle_len)
#             x_stance_rl = start + (end_stance_rl[0] - start) * (t_curr/T)
#             y_stance_rl = start + (end_stance_rl[1] - start) * (t_curr/T)

#             pd_leg = self.stance1 + np.array([x_swing_fl,  y_swing_fl, z_swing_fl,
#                                               x_stance_fr, y_stance_fr, 0.0,
#                                               x_stance_rl, y_stance_rl, 0.0,
#                                               x_swing_rr,  y_swing_rr, z_swing_rr])
            
#             return pd_leg

#         def double_stance2(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):

#             print('double stance 2')
            
#             # start by calculating the x values as an interpolation between the start and end
#             x_stance_fl = start + (end_fl[0] - start) * (t_curr/T)
#             x_stance_fr = start + (end_fr[0] - start) * (t_curr/T)
#             x_stance_rl = start + (end_rl[0] - start) * (t_curr/T)
#             x_stance_rr = start + (end_rr[0] - start) * (t_curr/T)

#             # calculate the y values as an interpolation between the start and end
#             y_stance_fl = start + (end_fl[1] - start) * (t_curr/T)
#             y_stance_fr = start + (end_fr[1] - start) * (t_curr/T)
#             y_stance_rl = start + (end_rl[1] - start) * (t_curr/T)
#             y_stance_rr = start + (end_rr[1] - start) * (t_curr/T)

#             pd_leg = self.double2_start + np.array([x_stance_fl, y_stance_fl, 0.0,
#                                                     x_stance_fr, y_stance_fr, 0.0,
#                                                     x_stance_rl, y_stance_rl, 0.0,
#                                                     x_stance_rr, y_stance_rr, 0.0,])
            
#             return pd_leg

#         def swingright_stanceleft(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):

#             print('swing right stance left')
            
#             # start by calculating the positions for the swing legs - fr, rl
#             x_swing_fr = start + (end_fr[0] - start) * (t_curr/T)
#             x_swing_rl = start + (end_rl[0] - start) * (t_curr/T)

#             y_swing_fr = start + (end_fr[1] - start) * (t_curr/T)
#             y_swing_rl = start + (end_rl[1] - start) * (t_curr/T)

#             # normalize the swing values to get the z-values from Bezier curves
#             x_swing_norm_fr = (x_swing_fr - 0)/(end_fr[0] - 0)
#             x_swing_norm_rl = (x_swing_rl - 0)/(end_rl[0] - 0)
            
#             bez_fr = Bezier3D(0.0, end_fr[0], self.p_stable[2], 0.1, 0.0, y_swing_fr)
#             z_swing_fr = bez_fr.create_bezier(x_swing_norm_fr)[-1]

#             bez_rl = Bezier3D(0.0, end_rl[0], self.p_stable[2], 0.1, 0.0, y_swing_rl)
#             z_swing_rl = bez_rl.create_bezier(x_swing_norm_rl)[-1]

#             # next calculate the positions for the stance legs - fl, rr
#             end_stance_fl = end_fl * (1 - (2 * self.double_stance_time)/self.cycle_len)
#             x_stance_fl = start + (end_stance_fl[0] - start) * (t_curr/T)
#             y_stance_fl = start + (end_stance_fl[1] - start) * (t_curr/T)

#             end_stance_rr = end_rr * (1 - (2 * self.double_stance_time)/self.cycle_len)
#             x_stance_rr = start + (end_stance_rr[0] - start) * (t_curr/T)
#             y_stance_rr = start + (end_stance_rr[1] - start) * (t_curr/T)

#             pd_leg = self.stance2 + np.array([x_stance_fl, y_stance_fl, 0.0,
#                                               x_swing_fr,  y_swing_fr, z_swing_fr,
#                                               x_swing_rl,  y_swing_rl, z_swing_rl,
#                                               x_stance_rr, y_stance_rr, 0.0])
            
#             return pd_leg
        
#         if t - self.cycle * self.cycle_len > self.cycle_len:
#             self.pd_leg_start = self.double1_start
#             self.cycle += 1
        
#         t = t % self.cycle_len   

#         if t > self.cycle_len:
#             pd_leg = self.pdlast     

#         if t < self.double_stance_time:

#             # in double stance mode

#             if self.cycle == 0:
#                 pd_leg = self.pdlast
#                 self.stance1 = pd_leg

#             else:

#                 # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
#                 v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
#                 v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
#                 v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
#                 v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

#                 # from these velocities calculate the ending foot positions, gives x and y
#                 pos_foot_fl = v_foot_fl * self.single_stance_time
#                 pos_foot_fr = v_foot_fr * self.single_stance_time
#                 pos_foot_rl = v_foot_rl * self.single_stance_time
#                 pos_foot_rr = v_foot_rr * self.single_stance_time

#                 end_fl = pos_foot_fl * (self.double_stance_time/self.cycle_len)
#                 end_fr = pos_foot_fr * (self.double_stance_time/self.cycle_len)
#                 end_rl = pos_foot_rl * (self.double_stance_time/self.cycle_len)
#                 end_rr = pos_foot_rr * (self.double_stance_time/self.cycle_len)
#                 pd_leg = double_stance1(t, self.double_stance_time, 0, end_fl, end_fr, end_rl, end_rr)
#                 self.stance1 = pd_leg
#             self.all_four_turn = True
        
#         elif t < (self.double_stance_time + self.single_stance_time):

#             # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
#             v_foot_fl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
#             v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
#             v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
#             v_foot_rr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

#             # from these velocities calculate the ending foot positions, gives x and y
#             pos_foot_fl = v_foot_fl * self.single_stance_time
#             pos_foot_fr = v_foot_fr * self.single_stance_time
#             pos_foot_rl = v_foot_rl * self.single_stance_time
#             pos_foot_rr = v_foot_rr * self.single_stance_time

#             # swing the left leg, stance the right leg
#             pd_leg = swingleft_stanceright(t - self.double_stance_time, self.single_stance_time, 0, pos_foot_fl, pos_foot_fr, pos_foot_rl, pos_foot_rr)
#             self.double2_start = pd_leg
#             self.all_four_turn = False

#         elif t < (2*self.double_stance_time + self.single_stance_time):

#             # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
#             v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
#             v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
#             v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
#             v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

#             # from these velocities calculate the ending foot positions, gives x and y
#             pos_foot_fl = v_foot_fl * self.single_stance_time
#             pos_foot_fr = v_foot_fr * self.single_stance_time
#             pos_foot_rl = v_foot_rl * self.single_stance_time
#             pos_foot_rr = v_foot_rr * self.single_stance_time


#             end_fl = pos_foot_fl * (self.double_stance_time/self.cycle_len)
#             end_fr = pos_foot_fr * (self.double_stance_time/self.cycle_len)
#             end_rl = pos_foot_rl * (self.double_stance_time/self.cycle_len)
#             end_rr = pos_foot_rr * (self.double_stance_time/self.cycle_len)
#             # both stance, re-adjust the body position
#             pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, end_fl, end_fr, end_rl, end_rr)
#             self.stance2 = pd_leg
#             self.all_four_turn = True
        
#         elif t < self.cycle_len:

#             # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
#             v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
#             v_foot_fr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
#             v_foot_rl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
#             v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

#             # from these velocities calculate the ending foot positions, gives x and y
#             pos_foot_fl = v_foot_fl * self.single_stance_time
#             pos_foot_fr = v_foot_fr * self.single_stance_time
#             pos_foot_rl = v_foot_rl * self.single_stance_time
#             pos_foot_rr = v_foot_rr * self.single_stance_time


#             pd_leg = swingright_stanceleft(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, pos_foot_fl, pos_foot_fr, pos_foot_rl, pos_foot_rr)
#             self.double1_start = pd_leg
#             self.all_four_turn = False

#         theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
#         theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
#         theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
#         theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

#         self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
#                       theta_FR[0], theta_FR[1], theta_FR[2],
#                       theta_RL[0], theta_RL[1], theta_RL[2],
#                       theta_RR[0], theta_RR[1], theta_RR[2]]
        
#         self.pdlast = pd_leg

#         # calculate the new base to foot values
#         self.fl_base_foot = self.chain_base_foot_fl.fkin(self.qcurr[self.fl[0]: self.fl[-1]+1])[0]
#         self.fr_base_foot = self.chain_base_foot_fr.fkin(self.qcurr[self.fr[0]: self.fr[-1]+1])[0]
#         self.rl_base_foot = self.chain_base_foot_rl.fkin(self.qcurr[self.rl[0]: self.rl[-1]+1])[0]
#         self.rr_base_foot = self.chain_base_foot_rr.fkin(self.qcurr[self.rr[0]: self.rr[-1]+1])[0]
        
        
#         return self.qcurr

#     def stabilize(self, t, T, z_comm):

#         # this is just going to do a stabilization such that the body is at a certain height and the feet are under the
#         # hips
#         p0_FLx = self.p0_foot_fl[0]
#         pf_FLx = self.p0_thigh_fl[0]


#         p0_FRx = self.p0_foot_fr[0]
#         pf_FRx = self.p0_thigh_fr[0]

#         p0_RLx = self.p0_foot_rl[0]
#         pf_RLx = self.p0_thigh_rl[0]

#         p0_RRx = self.p0_foot_rr[0]
#         pf_RRx = self.p0_thigh_rr[0]

#         # the desired leg z-position is just the negative of the z command
#         p0_FLz = self.p0_foot_fl[2]
#         pf_FLz = -z_comm

#         p0_FRz = self.p0_foot_fr[2]
#         pf_FRz = -z_comm

#         p0_RLz = self.p0_foot_rl[2]
#         pf_RLz = -z_comm

#         p0_RRz = self.p0_foot_rr[2]
#         pf_RRz = -z_comm

#         # thus we now calculate the current pf_curr as the interpolation between p0_FL and pf_FL

#         alph = t/T

#         pdes_x = (1 - alph) * p0_FLx + alph * pf_FLx
#         pdes_z = (1 - alph) * p0_FLz + alph * pf_FLz
#         pcurr_FL = [pdes_x, self.p0_foot_fl[1], pdes_z]

#         # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
#         theta_FL = NewtonRaphson(self.p0_foot_fl, pcurr_FL, self.qstable[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()


#         pdes_x = (1 - alph) * p0_FRx + alph * pf_FRx
#         pdes_z = (1 - alph) * p0_FRz + alph * pf_FRz
#         pcurr_FR = [pdes_x, self.p0_foot_fr[1], pdes_z]

#         # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
#         theta_FR = NewtonRaphson(self.p0_foot_fr, pcurr_FR, self.qstable[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        

#         pdes_x = (1 - alph) * p0_RLx + alph * pf_RLx
#         pdes_z = (1 - alph) * p0_RLz + alph * pf_RLz
#         pcurr_RL = [pdes_x, self.p0_foot_rl[1], pdes_z]

#         # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
#         theta_RL = NewtonRaphson(self.p0_foot_rl, pcurr_RL, self.qstable[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()

        
#         pdes_x = (1 - alph) * p0_RRx + alph * pf_RRx
#         pdes_z = (1 - alph) * p0_RRz + alph * pf_RRz
#         pcurr_RR = [pdes_x, self.p0_foot_rr[1], pdes_z]

#         # calculate the inverse kinematics by passing pcurr_FL, q_curr, and 
#         theta_RR = NewtonRaphson(self.p0_foot_rr, pcurr_RR, self.qstable[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson() 
        
#         # print(f'REG-IKIN theta_FL: {self.qstable[0:3]}')

#         self.qstable = [theta_FL[0], theta_FL[1], theta_FL[2],
#                         theta_FR[0], theta_FR[1], theta_FR[2],
#                         theta_RL[0], theta_RL[1], theta_RL[2],
#                         theta_RR[0], theta_RR[1], theta_RR[2]]
        


#         # also want to calcualate the position of the feet w.r.t. the base. will need for the new gait controller
#         self.fl_base_foot = self.chain_base_foot_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]+1])[0]
#         self.fr_base_foot = self.chain_base_foot_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]+1])[0]
#         self.rl_base_foot = self.chain_base_foot_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]+1])[0]
#         self.rr_base_foot = self.chain_base_foot_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]+1])[0]
        
#         self.qcurr = self.qstable

#         # print(f'alph: {alph}')
#         if round(alph, 1) == 1:
#             self.is_stable = True
#             _ = self.choose_gait_start()
        
#         return self.qstable
    
#     def walker(self, t, commands, errors, theta_curr):
        
#         if t < self.T_stab:
#             self.q_joints = self.stabilize(t, self.T_stab, self.zcomm)

#         else:
#             # goes into the walking controller
#             t_curr = t - self.T_stab
#             self.q_joints = self.walk_and_turn(t_curr, commands, theta_curr)

#         return self.q_joints

# if __name__=="__main__":

#     bot = QuadrupedController('./config/go2_config.yaml')
#     joints = bot.stand()
#     print(joints)




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
        self.cycle_forward = 0

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

        self.all_four_turn = False
        self.all_four_walk = False
        self.is_stable = False

        self.zcomm = 0.35
        self.mode = ''

        self.T_stab = 1.0
        self.T_reset = 0.35

        self.recovery_mode = False

        self.forward_reset_check = False

    # jointnames helper function
    def jointnames(self):
        # 12 joints
        return['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    
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
    
    def walk_and_turn(self, t, commands, yaw_curr):

        # NOTE: yaw_curr is supposed to be used for walking and turning
        
        def double_stance1(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):

            # start by calculating the x values as an interpolation between the start and end
            x_stance_fl = start + (end_fl[0] - start) * (t_curr/T)
            x_stance_fr = start + (end_fr[0] - start) * (t_curr/T)
            x_stance_rl = start + (end_rl[0] - start) * (t_curr/T)
            x_stance_rr = start + (end_rr[0] - start) * (t_curr/T)

            # calculate the y values as an interpolation between the start and end
            y_stance_fl = start + (end_fl[1] - start) * (t_curr/T)
            y_stance_fr = start + (end_fr[1] - start) * (t_curr/T)
            y_stance_rl = start + (end_rl[1] - start) * (t_curr/T)
            y_stance_rr = start + (end_rr[1] - start) * (t_curr/T)

            pd_leg = self.pd_leg_start + np.array([x_stance_fl, y_stance_fl, 0.0,
                                                   x_stance_fr, y_stance_fr, 0.0,
                                                   x_stance_rl, y_stance_rl, 0.0,
                                                   x_stance_rr, y_stance_rr, 0.0,])
            
            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):
            
            # start by calculating the positions for the swing legs - fl, rr
            x_swing_fl = start + (end_fl[0] - start) * (t_curr/T)
            x_swing_rr = start + (end_rr[0] - start) * (t_curr/T)

            y_swing_fl = start + (end_fl[1] - start) * (t_curr/T)
            y_swing_rr = start + (end_rr[1] - start) * (t_curr/T)

            # normalize the swing values to get the z-values from Bezier curves
            x_swing_norm_fl = (x_swing_fl - 0)/(end_fl[0] - 0)
            x_swing_norm_rr = (x_swing_rr - 0)/(end_rr[0] - 0)
            
            bez_fl = Bezier3D(0.0, end_fl[0], self.p_stable[2], 0.1, 0.0, y_swing_fl)
            z_swing_fl = bez_fl.create_bezier(x_swing_norm_fl)[-1]

            bez_rr = Bezier3D(0.0, end_rr[0], self.p_stable[2], 0.1, 0.0, y_swing_rr)
            z_swing_rr = bez_rr.create_bezier(x_swing_norm_rr)[-1]

            # next calculate the positions for the stance legs - fr, rl

            total_stance_time = 2 * self.double_stance_time + self.single_stance_time


            # end_stance_fr = end_fr * (1 - (2 * self.double_stance_time)/self.cycle_len)
            end_stance_fr = end_fr * self.single_stance_time/total_stance_time
            x_stance_fr = start + (end_stance_fr[0] - start) * (t_curr/T)
            y_stance_fr = start + (end_stance_fr[1] - start) * (t_curr/T)

            # end_stance_rl = end_rl * (1 - (2 * self.double_stance_time)/self.cycle_len)
            end_stance_rl = end_rl * self.single_stance_time/total_stance_time
            x_stance_rl = start + (end_stance_rl[0] - start) * (t_curr/T)
            y_stance_rl = start + (end_stance_rl[1] - start) * (t_curr/T)

            pd_leg = self.stance1 + np.array([x_swing_fl,  y_swing_fl, z_swing_fl,
                                              x_stance_fr, y_stance_fr, 0.0,
                                              x_stance_rl, y_stance_rl, 0.0,
                                              x_swing_rr,  y_swing_rr, z_swing_rr])
            
            return pd_leg

        def double_stance2(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):
            
            # start by calculating the x values as an interpolation between the start and end
            x_stance_fl = start + (end_fl[0] - start) * (t_curr/T)
            x_stance_fr = start + (end_fr[0] - start) * (t_curr/T)
            x_stance_rl = start + (end_rl[0] - start) * (t_curr/T)
            x_stance_rr = start + (end_rr[0] - start) * (t_curr/T)

            # calculate the y values as an interpolation between the start and end
            y_stance_fl = start + (end_fl[1] - start) * (t_curr/T)
            y_stance_fr = start + (end_fr[1] - start) * (t_curr/T)
            y_stance_rl = start + (end_rl[1] - start) * (t_curr/T)
            y_stance_rr = start + (end_rr[1] - start) * (t_curr/T)

            pd_leg = self.double2_start + np.array([x_stance_fl, y_stance_fl, 0.0,
                                                    x_stance_fr, y_stance_fr, 0.0,
                                                    x_stance_rl, y_stance_rl, 0.0,
                                                    x_stance_rr, y_stance_rr, 0.0,])
            
            return pd_leg

        def swingright_stanceleft(t_curr, T, start, end_fl, end_fr, end_rl, end_rr):
            
            # start by calculating the positions for the swing legs - fr, rl
            x_swing_fr = start + (end_fr[0] - start) * (t_curr/T)
            x_swing_rl = start + (end_rl[0] - start) * (t_curr/T)

            y_swing_fr = start + (end_fr[1] - start) * (t_curr/T)
            y_swing_rl = start + (end_rl[1] - start) * (t_curr/T)

            # normalize the swing values to get the z-values from Bezier curves
            x_swing_norm_fr = (x_swing_fr - 0)/(end_fr[0] - 0)
            x_swing_norm_rl = (x_swing_rl - 0)/(end_rl[0] - 0)
            
            bez_fr = Bezier3D(0.0, end_fr[0], self.p_stable[2], 0.1, 0.0, y_swing_fr)
            z_swing_fr = bez_fr.create_bezier(x_swing_norm_fr)[-1]

            bez_rl = Bezier3D(0.0, end_rl[0], self.p_stable[2], 0.1, 0.0, y_swing_rl)
            z_swing_rl = bez_rl.create_bezier(x_swing_norm_rl)[-1]

            total_stance_time = 2 * self.double_stance_time + self.single_stance_time

            # next calculate the positions for the stance legs - fl, rr
            # end_stance_fl = end_fl * (1 - (2 * self.double_stance_time)/self.cycle_len)
            end_stance_fl = end_fl * self.single_stance_time/total_stance_time
            x_stance_fl = start + (end_stance_fl[0] - start) * (t_curr/T)
            y_stance_fl = start + (end_stance_fl[1] - start) * (t_curr/T)

            # end_stance_rr = end_rr * (1 - (2 * self.double_stance_time)/self.cycle_len)
            end_stance_rr = end_rr * self.single_stance_time/total_stance_time
            x_stance_rr = start + (end_stance_rr[0] - start) * (t_curr/T)
            y_stance_rr = start + (end_stance_rr[1] - start) * (t_curr/T)

            pd_leg = self.stance2 + np.array([x_stance_fl, y_stance_fl, 0.0,
                                              x_swing_fr,  y_swing_fr, z_swing_fr,
                                              x_swing_rl,  y_swing_rl, z_swing_rl,
                                              x_stance_rr, y_stance_rr, 0.0])
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:
            self.pd_leg_start = self.double1_start
            self.pd_leg_start[2::3] = self.p_stable[2::3]
            self.cycle += 1
        
        t = t % self.cycle_len     

        if t < self.double_stance_time:

            # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
            v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
            v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
            v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
            v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

            # from these velocities calculate the ending foot positions, gives x and y
            pos_foot_fl = v_foot_fl * self.single_stance_time
            pos_foot_fr = v_foot_fr * self.single_stance_time
            pos_foot_rl = v_foot_rl * self.single_stance_time
            pos_foot_rr = v_foot_rr * self.single_stance_time

            total_stance_time = 2 * self.double_stance_time + self.single_stance_time

            end_fl = pos_foot_fl * (self.double_stance_time/total_stance_time)
            end_fr = pos_foot_fr * (self.double_stance_time/total_stance_time)
            end_rl = pos_foot_rl * (self.double_stance_time/total_stance_time)
            end_rr = pos_foot_rr * (self.double_stance_time/total_stance_time)

            pd_leg = double_stance1(t, self.double_stance_time, 0, end_fl, end_fr, end_rl, end_rr)
            self.stance1 = pd_leg
            self.all_four_turn = True
        
        elif t < (self.double_stance_time + self.single_stance_time):

            # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
            # fl, rr in swing - fr, rl in stance
            v_foot_fl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
            v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
            v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
            v_foot_rr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

            # from these velocities calculate the ending foot positions, gives x and y
            pos_foot_fl = v_foot_fl * self.single_stance_time
            pos_foot_fr = v_foot_fr * self.single_stance_time
            pos_foot_rl = v_foot_rl * self.single_stance_time
            pos_foot_rr = v_foot_rr * self.single_stance_time

            # swing the left leg, stance the right leg
            pd_leg = swingleft_stanceright(t - self.double_stance_time, self.single_stance_time, 0, pos_foot_fl, pos_foot_fr, pos_foot_rl, pos_foot_rr)
            self.double2_start = pd_leg
            self.all_four_turn = False

        elif t < (2*self.double_stance_time + self.single_stance_time):

            # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
            v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
            v_foot_fr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
            v_foot_rl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
            v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

            # from these velocities calculate the ending foot positions, gives x and y
            pos_foot_fl = v_foot_fl * self.single_stance_time
            pos_foot_fr = v_foot_fr * self.single_stance_time
            pos_foot_rl = v_foot_rl * self.single_stance_time
            pos_foot_rr = v_foot_rr * self.single_stance_time


            total_stance_time = 2 * self.double_stance_time + self.single_stance_time

            end_fl = pos_foot_fl * (self.double_stance_time/total_stance_time)
            end_fr = pos_foot_fr * (self.double_stance_time/total_stance_time)
            end_rl = pos_foot_rl * (self.double_stance_time/total_stance_time)
            end_rr = pos_foot_rr * (self.double_stance_time/total_stance_time)

            # both stance, re-adjust the body position
            pd_leg = double_stance2(t - (self.double_stance_time + self.single_stance_time), self.double_stance_time, 0, end_fl, end_fr, end_rl, end_rr)
            self.stance2 = pd_leg
            self.all_four_turn = True
        
        elif t < self.cycle_len:

            # calculate the velocities for fl, fr, rl, rr, only want the x and y velocities
            v_foot_fl = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.fl_base_foot))[0:2]
            v_foot_fr = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.fr_base_foot))[0:2]
            v_foot_rl = commands[0:2] + (np.cross(np.array([0, 0, commands[2]]), self.rl_base_foot))[0:2]
            v_foot_rr = -commands[0:2] - (np.cross(np.array([0, 0, commands[2]]), self.rr_base_foot))[0:2]

            # from these velocities calculate the ending foot positions, gives x and y
            pos_foot_fl = v_foot_fl * self.single_stance_time
            pos_foot_fr = v_foot_fr * self.single_stance_time
            pos_foot_rl = v_foot_rl * self.single_stance_time
            pos_foot_rr = v_foot_rr * self.single_stance_time


            pd_leg = swingright_stanceleft(t - (2*self.double_stance_time+self.single_stance_time), self.single_stance_time, 0, pos_foot_fl, pos_foot_fr, pos_foot_rl, pos_foot_rr)
            self.double1_start = pd_leg
            self.all_four_turn = False


        rot_mat = np.array([[cos(yaw_curr), -sin(yaw_curr), 0.0],
                            [sin(yaw_curr), cos(yaw_curr), 0.0],
                            [0.0, 0.0, 1.0]])

        # v_base_y_curr = (np.transpose(rot_mat) @ self.data.qvel[self.mj_idx.v_base_vel_idx])[1]

        v_base_y_curr = (rot_mat.T @ self.data.qvel[self.mj_idx.v_base_vel_idx])[1]

        print(f'v_base_y_curr: {v_base_y_curr}')

        # print(f'current y-velocity: {v_base_y_curr}')
        # print(f'current y-position: {self.data.qvel[self.mj_idx.POS_Y]}')
        v_base_y_desired = 0.0

        kv = -0.06 * commands[0]/0.8

        pd_leg[1] = pd_leg[1] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[4] = pd_leg[4] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[7] = pd_leg[7] - kv*(v_base_y_curr - v_base_y_desired)
        pd_leg[10] = pd_leg[10] - kv*(v_base_y_curr - v_base_y_desired)
        

        theta_FL = NewtonRaphson(self.pdlast[0:3], pd_leg[0:3], self.qcurr[0:3], self.chain_base_foot_fl, self.chain_base_hip_fl).call_newton_raphson()
        theta_FR = NewtonRaphson(self.pdlast[3:6], pd_leg[3:6], self.qcurr[3:6], self.chain_base_foot_fr, self.chain_base_hip_fr).call_newton_raphson()
        theta_RL = NewtonRaphson(self.pdlast[6:9], pd_leg[6:9], self.qcurr[6:9], self.chain_base_foot_rl, self.chain_base_hip_rl).call_newton_raphson()
        theta_RR = NewtonRaphson(self.pdlast[9:12], pd_leg[9:12], self.qcurr[9:12], self.chain_base_foot_rr, self.chain_base_hip_rr).call_newton_raphson()

        self.qcurr = [theta_FL[0], theta_FL[1], theta_FL[2],
                      theta_FR[0], theta_FR[1], theta_FR[2],
                      theta_RL[0], theta_RL[1], theta_RL[2],
                      theta_RR[0], theta_RR[1], theta_RR[2]]
        
        self.pdlast = pd_leg

        # calculate the new base to foot values
        self.fl_base_foot = self.chain_base_foot_fl.fkin(self.qcurr[self.fl[0]: self.fl[-1]+1])[0]
        self.fr_base_foot = self.chain_base_foot_fr.fkin(self.qcurr[self.fr[0]: self.fr[-1]+1])[0]
        self.rl_base_foot = self.chain_base_foot_rl.fkin(self.qcurr[self.rl[0]: self.rl[-1]+1])[0]
        self.rr_base_foot = self.chain_base_foot_rr.fkin(self.qcurr[self.rr[0]: self.rr[-1]+1])[0]
        
        
        return self.qcurr

    def stabilize(self, t, T, z_comm):

        self.deltx = 0.6 * (self.single_stance_time - self.double_stance_time)

        # this is just going to do a stabilization such that the body is at a certain height and the feet are under the
        # hips
        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0] - self.deltx/2


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0] + self.deltx/2

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0] + self.deltx/2

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0] - self.deltx/2

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
        


        # also want to calcualate the position of the feet w.r.t. the base. will need for the new gait controller
        self.fl_base_foot = self.chain_base_foot_fl.fkin(self.qstable[self.fl[0]: self.fl[-1]+1])[0]
        self.fr_base_foot = self.chain_base_foot_fr.fkin(self.qstable[self.fr[0]: self.fr[-1]+1])[0]
        self.rl_base_foot = self.chain_base_foot_rl.fkin(self.qstable[self.rl[0]: self.rl[-1]+1])[0]
        self.rr_base_foot = self.chain_base_foot_rr.fkin(self.qstable[self.rr[0]: self.rr[-1]+1])[0]
        
        self.qcurr = self.qstable

        # print(f'alph: {alph}')
        if round(alph, 1) == 1:
            self.is_stable = True
            _ = self.choose_gait_start()
        
        return self.qstable
    
    def walker(self, t, commands, errors, theta_curr):

        goal_found = False

        print(f'commands: {commands}')
        print(f'errors: {errors}')
        
        if t < self.T_stab:
            self.q_joints = self.stabilize(t, self.T_stab, self.zcomm)

        else:
            # goes into the walking controller
            t_curr = t - self.T_stab
            self.q_joints = self.walk_and_turn(t_curr, commands, theta_curr)

            if errors[0] <= 0.01 and errors[-1] <= 1e-1:
                goal_found = True

        return self.q_joints, goal_found

if __name__=="__main__":

    bot = QuadrupedController('./config/go2_config.yaml')
    joints = bot.stand()
    print(joints)
