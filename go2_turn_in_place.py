# Features
# DIAGONAL trotting
# Starts from some user-defined stable configuration
# Newton-Raphson Method for Inverse Kinematics

#!/usr/bin/env python3.10

import mujoco
import glfw

import numpy as np
import yaml
import time
from math import acos, atan2, sqrt, sin, cos

from definitions.go2_definitions import Mujoco_IDX_go2
from definitions.KinematicChain import KinematicChain
from definitions.NewtonRaphson import NewtonRaphson
from definitions.Bezier3D import Bezier3D

from hw5code.TransformHelpers import *

SEMI_MIN = 0.00194
SEMI_MAJ = 0.0054

# define the Trajectory class
class Trajectory():
    # Initialize stuff

    def __init__(self):

        # create helper function to create index lists for joints
        def indexlist(start, num): return list(range(start, start+num))

        # initialize the joint indices
        self.fl = [0, 1, 2]
        self.fr = [3, 4, 5]
        self.rl = [6, 7, 8]
        self.rr = [9, 10, 11]

        # TODO: Define the kinematice chain

        
        # define the initial joint positions
        q = data.qpos
        self.q_joints = q[mj_idx.q_joint_idx]
        self.qstable = self.q_joints
        self.qcurr = self.qstable

        self.cycle = 0

        self.urdf = './models/go2/go2.urdf'

        # set up the kinematic chains for each leg

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

        self.cycle_len = 0.5

        self.single_stance_time = 3 * self.cycle_len/8
        self.double_stance_time = self.cycle_len/8

        # recall that we found a reasonable deltax distance by looking at the 
        # desired speed of the body and the time taken for one foot swing.

        # similarly, if we take the rotation speed e.g. 0.1 radians per second
        # then we can calculate delta theta as follows
        self.rot_speed = 0.5
        self.delta_theta = self.rot_speed * self.single_stance_time

        print(f'self.delta_theta: {self.delta_theta}')

        self.r = sqrt((0.1934)**2 + (0.0955+0.0465)**2)

        print(f'self.r: {self.r}')

    # define the names of the joints
    def jointnames(self):
        # 12 joints
        return['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
               'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
               'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
               'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    def goto(self, t, T, p0, pf):

        p = p0 + (pf - p0) * (4*(t/T) - 4*(t/T)**2)
        v = (pf - p0) * (4/T - 8*(t/T**2))

        return (p, v)
    
    def stabilize(self, t, T, deltax, z_comm): # assume deltax is zero
        

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

    def evaluate(self, t, dt):

        # hard-coding everything for now
        
        # get the radii for the legs from the center

        # get the distances
        # FL: hip to thigh: 0.0955 in y
        # FL: base to hip: 0.1934 in x, 0.0465 in y

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

            # print(f'[x_swing, y_swing, z_swing]: {[x_swing, y_swing, z_swing]}')
            
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
        
        self.pd_last = pd_leg
        
        return self.qcurr


# main function to run the system
if __name__ == '__main__':

    # load the config file
    config_file = './config/go2_config.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # load the mujoco model
    mj_model_path = config['MODEL']['xml_path']
    model = mujoco.MjModel.from_xml_path(mj_model_path)
    data = mujoco.MjData(model)

    # setup the glfw window
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(1920, 1080, "Robot", None, None)
    glfw.make_context_current(window)

    # set the window to be resizable
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # create camera to render the scene
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    cam.distance = 2.0
    cam.elevation = -25
    cam.azimuth = 90

    # turn on reaction forces
    if config['VISUALIZATION']['grf']:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # create the scene and cont
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)

    # create the index objects
    mj_idx = Mujoco_IDX_go2()

    # gains
    kp = np.array(config['GAINS']['kps'])
    kd = np.array(config['GAINS']['kds'])

    # default joint positions
    q0_joint = np.array(config['INITIAL_STATE']['default_angles'])
    v0_joint = np.zeros(len(q0_joint))

    # set the initial state
    data.qpos = np.zeros(model.nq)
    data.qvel = np.zeros(model.nv)
    data.qpos[mj_idx.q_base_pos_idx[0]] = config['INITIAL_STATE']['px']
    data.qpos[mj_idx.q_base_pos_idx[1]] = config['INITIAL_STATE']['py']
    data.qpos[mj_idx.q_base_pos_idx[2]] = config['INITIAL_STATE']['pz']
    data.qpos[mj_idx.q_base_quat_idx[0]] = 1.0
    data.qpos[mj_idx.q_joint_idx] = q0_joint
    data.qvel[mj_idx.v_joint_idx] = v0_joint

    # timers
    t_sim = 0.0
    dt_sim = model.opt.timestep
    max_sim_time = config['SIM']['t_max']

    # for rendering frequency
    hz_render = config['SIM']['hz_render']
    dt_render = 1.0 / hz_render
    counter = 0
    t1_wall = 0.0
    t2 = 0.0
    dt_wall = 0.0

    # for control frequency
    hz_control = config['SIM']['hz_control']
    dt_control = 1.0 / hz_control
    decimation = int(dt_control / dt_sim)

    # for timing the total sim time
    t0_real_time = time.time()

    fact = 0.0

    # define the trajectory object
    traj = Trajectory()

    q = data.qpos
    v = data.qvel

    q_joints = q[mj_idx.q_joint_idx]
    v_joints = v[mj_idx.v_joint_idx]

    q_joints_des = np.zeros_like(q_joints)

    t0 = 0.5
    T_stab = 0.5
    deltx = 0.0
    z_command = 0.35

    # main simulation loop
    while (not glfw.window_should_close(window)) and (t_sim < max_sim_time):

        t0_sim = data.time

        # t_sim is the total/running simulation
        while (t_sim - t0_sim) < (dt_render):

            # print(f'Current z height is: {data.qpos[mj_idx.POS_Z]}')

            t1_wall = time.time()
            t_sim = data.time

            if counter % decimation == 0:

                # generslized position
                q = data.qpos

                # generslized velocity
                v = data.qvel

                q_joints = q[mj_idx.q_joint_idx]
                v_joints = v[mj_idx.v_joint_idx]
                v_joints_des = np.zeros_like(v_joints)

                if t_sim <= t0:
                    # this is the time for which the robot stabilizes (using the yaml data) after being
                    # dropped into the simulation
                    q_joints_des = q0_joint

                elif t_sim > t0 and (t_sim - t0) <= T_stab:

                    # now we're in the actual control phase
                    t_stabx = t_sim - t0 # this is the actual time elapsed during the control phase

                    # we should probably define a separate stabilization function for it to just get into a good position
                    
                    # so here we send over the current time t_stab and the time for which I want it to perform the stabilization
                    # for T_stab. Also send over the desired deltax (in this case 0) and some desired initial z position

                    q_joints_des = traj.stabilize(t_stabx, T_stab, deltx, z_command)

                    traj.p_stable = traj.choose_gait_start()

                else:

                    t_curr = t_sim - (t0 + T_stab)

                    q_joints_des = traj.evaluate(t_curr, dt_sim)

                # compute torque from PID
                u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

            # set the control torque
            data.ctrl[:] = u

            # advance the simulation
            mujoco.mj_step(model, data)

            # wait until next sim update
            t2_wall = time.time()
            dt_wall = t2_wall - t1_wall
            dt_sleep = dt_sim - dt_wall
            if dt_sleep > 0.0:
                time.sleep(dt_sleep)

        px_base = data.qpos[mj_idx.q_base_pos_idx[0]]
        py_base = data.qpos[mj_idx.q_base_pos_idx[1]]
        pz_base = data.qpos[mj_idx.q_base_pos_idx[2]]
        cam.lookat[0] = px_base
        cam.lookat[1] = py_base
        cam.lookat[2] = 0.4

        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        mujoco.mjr_render(viewport, scene, context)

        t1_real_time = time.time()
        label_text = f"Sim Time: {t_sim:.2f} sec \nWall Time: {t1_real_time - t0_real_time:.3f} sec"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_200,    # font scale
            mujoco.mjtGridPos.mjGRID_TOPLEFT,       # position on screen
            viewport,                               # this must be the MjrRect, not context
            label_text,                             # main overlay text (string, not bytes)
            "",                                     # optional secondary text
            context
        )

        glfw.swap_buffers(window)

        glfw.poll_events()

    print(f"Total simulation time: {t1_real_time - t0_real_time:.3f} seconds")
