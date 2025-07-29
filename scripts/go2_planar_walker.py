#!/usr/bin/env python3.10

import mujoco
import glfw

import numpy as np
import yaml
import time
from math import acos, atan2, sqrt, sin, cos

from definitions.go2_definitions import Mujoco_IDX_go2
from definitions.KinematicChain import KinematicChain

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

        # TODO: Define the kinematice chain

        
        # define the initial joint positions
        self.q0 = [0.0, 0.0, -0.84,
                   0.0, 0.0, -0.84,
                   0.0, 0.0, -0.84,
                   0.0, 0.0, -0.84,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0]
        
        # get the local position (with respect to the hip)
        self.p0 = np.array([0, 0, -0.426,
                            0, 0, -0.426,
                            0, 0, -0.426,
                            0, 0, -0.426])

        self.cycle = 0

        self.urdf = './models/go2/go2.urdf'

        # set up the kinematic chains for each leg
        self.chain_fl = KinematicChain('base', 'FL_foot', self.jointnames()[0:3], self.urdf)
        self.chain_fr = KinematicChain('base', 'FR_foot', self.jointnames()[3:6], self.urdf)
        self.chain_rl = KinematicChain('base', 'RL_foot', self.jointnames()[6:9], self.urdf)
        self.chain_rr = KinematicChain('base', 'RR_foot', self.jointnames()[9:12], self.urdf)


        # initialize the position of the body
        self.pbody = np.array([0.0, 0.0, 0.355])

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

        p = p0 + (pf - p0) * (3*(t/T)**2 - 2*(t/T)**3)
        v = (pf - p0)/T * (6 * (t/T) - 6 * (t/T)**2)

        return (p, v)
    
    # calculate the robot's position given the current t value
    def evaluate(self, t, dt):

        inc_up = np.array([0, 0, -0.1,
                           0, 0, -0.1,
                           0, 0, -0.1,
                           0, 0, -0.1])

        inc_down = np.array([0, 0, 0.1,
                           0, 0, 0.1,
                           0, 0, 0.1,
                           0, 0, 0.1])
        
    
        # take the current time step and use this to determine what step in the
        # cycle we're at

        self.goHome = self.p0
        self.goUp = self.p0 + inc_up
        self.goDown = self.p0 + inc_down


        # calculate the desired foot positions
        t = t % 0.3

        if t < 0.15:

            (s0, s0_dot) = self.goto(t, 0.3, 0.0, 1.0)

            pd_leg = self.p0 + (self.goUp - self.p0) * s0

        elif t < 0.3:

            (s0, s0_dot) = self.goto(t-0.15, 0.3, 1.0, 0.0)

            pd_leg = self.goUp + (self.goHome - self.goUp) * s0

        
        pd_FL = pd_leg[0:3]

        print(f'\npd_FL is {pd_FL}')

        r_FL = sqrt((pd_leg[0] - self.p0[0])**2 + (pd_leg[2] - self.p0[2])**2)

        theta0_FL = 0.0
        theta2_FL = acos( (r_FL**2 - self.l_thigh**2 - self.l_calf**2)/(2*self.l_thigh*self.l_calf))
        theta1_FL = atan2((pd_leg[2] - self.p0[2]), (pd_leg[0] - self.p0[0])) - atan2(self.l_calf*sin(theta2_FL), self.l_thigh + self.l_calf*cos(theta2_FL))
   
        qd = np.array([theta0_FL, theta1_FL, theta2_FL,
                       theta0_FL, theta1_FL, theta2_FL,
                       theta0_FL, theta1_FL, theta2_FL,
                       theta0_FL, theta1_FL, theta2_FL])

        return qd
    
        
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
    cam.elevation = -15
    cam.azimuth = 150

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

    # main simulation loop
    while (not glfw.window_should_close(window)) and (t_sim < max_sim_time):

        t0_sim = data.time

        while (t_sim - t0_sim) < (dt_render):

            t1_wall = time.time()
            t_sim = data.time

            if counter % decimation == 0:

                q = data.qpos
                v = data.qvel

                q_joints = q[mj_idx.q_joint_idx]
                v_joints = v[mj_idx.v_joint_idx]

                print(f'\ncurrent q_joints: {q_joints}')

                print(f'\nError is {np.subtract(q_joints, q_joints_des)}')
                time.sleep(5)


                # enter the controller block(?) by calling the evaluate function on the object
                q_joints_des = traj.evaluate(t0_sim, dt_sim)

                # print(f'q_joints_des are {q_joints_des}')
                    

                # compute torque from PID
                # u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

                u = -600 * (q_joints - q_joints_des)


            # set the control torque
            data.ctrl[:] = u

            data.qpos[0] = 0
            data.qpos[1] = 0
            data.qpos[2] = 1
            data.qpos[3] = 1
            data.qpos[4] = 0
            data.qpos[5] = 0
            data.qpos[6] = 0

            data.qvel[0] = 0
            data.qvel[1] = 0
            data.qvel[2] = 0
            data.qvel[3] = 0
            data.qvel[4] = 0
            data.qvel[5] = 0


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
        cam.lookat[2] = pz_base

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
