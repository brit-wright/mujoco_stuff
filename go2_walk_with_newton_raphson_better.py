# Features
# WALKING FORWARD
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

from hw5code.TransformHelpers import *


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

        self.x_list = []
        self.t_list = []
        self.z_list = []

        # TODO: Define the kinematice chain

        
        # define the initial joint positions
        q = data.qpos
        self.q_joints = q[mj_idx.q_joint_idx]
        self.qstable = self.q_joints
        self.qcurr = self.qstable

        self.cycle = 0

        self.urdf = './models/go2/go2.urdf'

        # set up the kinematic chains for each leg

        self.cycle_len = 0.4

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

    # define the names of the joints
    def jointnames(self):
        # 12 joints
        return['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
               'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
               'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
               'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    def goto(self, t, T, p0, pf):

        # p = p0 + (pf - p0) * (4*(t/T) - 4*(t/T)**2)
        # v = (pf - p0) * (4/T - 8*(t/T**2))

        p = p0 + (pf-p0)   * (3*(t/T)**2 - 2*(t/T)**3)
        v =    + (pf-p0)/T * (6*(t/T)    - 6*(t/T)**2)

        return (p, v)
    
    def stabilize(self, t, T, z_comm, vx): # assume deltax is zero
        
        self.deltx = vx * self.cycle_len/4

        p0_FLx = self.p0_foot_fl[0]
        pf_FLx = self.p0_thigh_fl[0] + self.deltx/2


        p0_FRx = self.p0_foot_fr[0]
        pf_FRx = self.p0_thigh_fr[0] - self.deltx/2

        p0_RLx = self.p0_foot_rl[0]
        pf_RLx = self.p0_thigh_rl[0] - self.deltx/2

        p0_RRx = self.p0_foot_rr[0]
        pf_RRx = self.p0_thigh_rr[0] + self.deltx/2

        # p0_FLx = self.p0_foot_fl[0]
        # pf_FLx = self.p0_thigh_fl[0] + 0.1


        # p0_FRx = self.p0_foot_fr[0]
        # pf_FRx = self.p0_thigh_fr[0] - 0.1

        # p0_RLx = self.p0_foot_rl[0]
        # pf_RLx = self.p0_thigh_rl[0] - 0.1

        # p0_RRx = self.p0_foot_rr[0]
        # pf_RRx = self.p0_thigh_rr[0] + 0.1

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
        
        return self.p_stable
    
    def evaluate(self, t, dt):

        # Note: every cycle starts off with the front left forward

        # print(f'self.deltx: {self.deltx}')

        def double_stance1():
            
            pd_leg = self.pdlast

            self.stance1 = pd_leg

            return pd_leg

        def stanceleft_swingright(t_curr, T, start, end):
            
            # need to calculate pdleg based on location in phase we're at

            # first determine x location using linear interpolation
            x_curr = start + (end - start) * (t_curr/T)

            # now we determing the z location by defining a parabola
            z_curr = -x_curr * (x_curr - T) * 30

            # now we adjust pd_leg based on these values
            pd_leg = self.stance1 + np.array([-x_curr, 0.0, 0.0,
                                x_curr, 0.0, z_curr,
                                x_curr, 0.0, z_curr,
                                -x_curr, 0.0, 0.0])
            
            return pd_leg

        def double_stance2():
            
            pd_leg = self.pdlast

            self.stance2 = pd_leg

            return pd_leg

        def swingleft_stanceright(t_curr, T, start, end):
            
            # need to calculate pdleg based on location in phase we're at

            # first determine x location using linear interpolation
            x_curr = start + (end - start) * (t_curr/T)

            # now we determing the z location by defining a parabola
            
            z_curr = -x_curr * (x_curr - T) * 30

            # now we adjust pd_leg based on these values
            pd_leg = self.stance2 + np.array([x_curr, 0.0, z_curr,
                                -x_curr, 0.0, 0.0,
                                -x_curr, 0.0, 0.0,
                                x_curr, 0.0, z_curr])
            
            print(f'pd_leg is: {pd_leg}')
            print(f'x_curr is: {x_curr}')
            
            return pd_leg
        
        if t - self.cycle * self.cycle_len > self.cycle_len:

            # print(f'checking x-positions: {self.pdlast[0]}, {self.pdlast[3]}, {self.pdlast[6]}, {self.pdlast[9]}')
            self.cycle += 1

        t = t % self.cycle_len

        if t < 0.25 * self.cycle_len:
            # print('phase1')
            pd_leg = double_stance1()

        elif t < 0.5 * self.cycle_len:
            # print('phase2')
            pd_leg = stanceleft_swingright(t - 0.25*self.cycle_len, 0.25*self.cycle_len, 0, self.deltx)

        elif t < 0.75 * self.cycle_len:
            
            # print('phase3')
            pd_leg = double_stance2()

        elif t < self.cycle_len:

            # print('phase4')
            pd_leg = swingleft_stanceright(t - 0.75*self.cycle_len, 0.25*self.cycle_len, 0, self.deltx)

        else:
            print('nothing satisfied')

        # print(f't is {t}')
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
    cam.elevation = -1
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
    v_bod = 1

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

                    q_joints_des = traj.stabilize(t_stabx, T_stab, z_command, v_bod)

                    traj.p_stable = traj.choose_gait_start()

                else:

                    t_curr = t_sim - (t0 + T_stab)

                    q_joints_des = traj.evaluate(t_curr, dt_sim)

                # compute torque from PID
                u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

            # set the control torque
            data.ctrl[:] = u

            # data.qpos[0] = 0
            # data.qpos[1] = 0
            # data.qpos[2] = 1 #this
            # data.qpos[3] = 1
            # data.qpos[4] = 0
            # data.qpos[5] = 0
            # data.qpos[6] = 0

            # data.qvel[0] = 0
            # data.qvel[1] = 0
            # data.qvel[2] = 0 # this
            # data.qvel[3] = 0
            # data.qvel[4] = 0
            # data.qvel[5] = 0 

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
