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

Q0 = [0.0, 0.35, -0.84,
                   0.0, 0.35, -0.84,
                   0.0, 0.35, -0.84,
                   0.0, 0.35, -0.84,]

ORIGIN = [0.0, 0.0]
LEN_STRETCHED = 0.432

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
        self.q0 = [0.0, 0.50, -1.66,
                   0.0, 0.50, -1.66,
                   0.0, 0.50, -1.66,
                   0.0, 0.50, -1.66,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0]
        
        self.qd = [0.0, 0.40, 0.0,
                   0.0, 0.40, 0.0,
                   0.0, 0.40, 0.0,
                   0.0, 0.40, 0.0]

        self.cycle = 0

        self.urdf = './models/go2/go2.urdf'

        # set up the kinematic chains for each leg
        self.chain_fl = KinematicChain('base', 'FL_foot', self.jointnames()[0:3], self.urdf)
        self.chain_fr = KinematicChain('base', 'FR_foot', self.jointnames()[3:6], self.urdf)
        self.chain_rl = KinematicChain('base', 'RL_foot', self.jointnames()[6:9], self.urdf)
        self.chain_rr = KinematicChain('base', 'RR_foot', self.jointnames()[9:12], self.urdf)

        self.chain_thigh_FL = KinematicChain('base', 'FL_thigh', ['FL_hip_joint', 'FL_thigh_joint'], self.urdf)
        self.chain_thigh_FR = KinematicChain('base', 'FR_thigh', ['FR_hip_joint', 'FR_thigh_joint'], self.urdf)
        self.chain_thigh_RL = KinematicChain('base', 'RL_thigh', ['RL_hip_joint', 'RL_thigh_joint'], self.urdf)
        self.chain_thigh_RR = KinematicChain('base', 'RR_thigh', ['RR_hip_joint', 'RR_thigh_joint'], self.urdf)

        self.chain_calf_FL = KinematicChain('base', 'FL_calf', ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'], self.urdf)


        (self.p_interest1, R_1, b, c) = self.chain_fl.fkin(self.q0[self.fl[0]: self.fl[-1]+1])
        (self.p_interest2, R_2, b, c) = self.chain_fr.fkin(self.q0[self.fr[0]: self.fr[-1]+1])
        (self.p_interest3, R_3, b, c) = self.chain_rl.fkin(self.q0[self.rl[0]: self.rl[-1]+1])
        (self.p_interest4, R_4, b, c) = self.chain_rr.fkin(self.q0[self.rr[0]: self.rr[-1]+1])

        (self.p_thigh_FL, self.R_thigh, b, c) = self.chain_thigh_FL.fkin([self.q0[0], self.q0[1]])
        (self.p_thigh_FR, self.R_thigh, b, c) = self.chain_thigh_FR.fkin([self.q0[3], self.q0[4]])
        (self.p_thigh_RL, self.R_thigh, b, c) = self.chain_thigh_RL.fkin([self.q0[6], self.q0[7]])
        (self.p_thigh_RR, self.R_thigh, b, c) = self.chain_thigh_RR.fkin([self.q0[9], self.q0[10]])


        (self.p_calf_FL, self.R_calf, b, c) = self.chain_calf_FL.fkin([self.q0[0], self.q0[1], self.q0[2]])


        self.p0 = np.concatenate((self.p_interest1, self.p_interest2, self.p_interest3, self.p_interest4))
        
        # initialize the position of the body
        self.pbody = np.array([0.0, 0.0, 0.355])

        self.l_thigh = 0.213
        self.l_calf = 0.213

        self.x_list = []
        self.z_list = []

        self.x_curr = []
        self.z_curr = []

        self.num_cycles = 0
        self.num_iters = 0


        self.pd_last = self.p0

    def calculate_distance(self, point1, point2):
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def check_point(self, point):
        # we essentially check that the point is not equal to the origin and that the distance of the point from
        # the origin is not greater than the length of the stretched leg
        
        if point[0] == ORIGIN[0] and point[1] == ORIGIN[1]:
            return False
        
        else:
            dist1 = self.calculate_distance(ORIGIN, point)
            
            if dist1 > LEN_STRETCHED:
                return False
            
        return True


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

        self.num_iters += 1

        cycle_len = 2*np.pi

        # check whether one step cycle has finished or not
        if t - self.cycle*cycle_len > cycle_len:
        
            self.num_cycles += 1

        t = t % cycle_len

        # okay let's draw a circle. let's seeee

        # to draw the circle i want to cycle through angles from 0 to 360 (0 to 2pi)
        # also want to define the radial distance. for now let's call it 0.1

        # set the desired leg position as x = 0.1cos(theta), z = 0.1sin(theta)
        r = 0.01

        frequency = 2.0
        omega = 2 *np.pi * frequency
        th = omega * t

        pd_leg_inc = [r*cos(th), 0.0, r*sin(th),
                         r*cos(th), 0.0, r*sin(th),
                         r*cos(th), 0.0, r*sin(th),
                         r*cos(th), 0.0, r*sin(th)]
        
        pd_leg = self.pd_last + pd_leg_inc


        # print(f'pd_leg: {pd_leg}')
        if self.num_iters % 20 == 0:
            self.x_list.append(pd_leg[0])
            self.z_list.append(pd_leg[2])

        #     (p_curr, _, _, _) = self.chain_fl.fkin(self.qd[self.fl[0]: self.fl[-1]+1])
        #     self.z_curr.append(p_curr[2])
        #     self.x_curr.append(p_curr[0])
        
        theta0_FL = 0.0
        toe_pos = pd_leg[0:3] # from the frame of the base
        thigh_pos = self.p_thigh_FL # from the frame of the base
        r_local_FL = toe_pos - thigh_pos # the vector pointing from the thigh to the toe
        x_local = r_local_FL[0]
        z_local = r_local_FL[2]
        r_FL = sqrt(x_local**2 + z_local**2)
        cos_theta2 = (r_FL**2 - self.l_thigh**2 - self.l_calf**2)/(2*self.l_thigh*self.l_calf)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2_FL = -acos(cos_theta2) # is always negative
        alpha = atan2(x_local, z_local)
        beta = atan2(self.l_calf*abs(sin(theta2_FL)), -(self.l_thigh+self.l_calf*cos(theta2_FL)))
        theta1_FL = alpha - beta

        theta0_FR = 0.0
        toe_pos = pd_leg[3:6] # from the frame of the base
        thigh_pos = self.p_thigh_FR # from the frame of the base
        r_local_FR = toe_pos - thigh_pos # the vector pointing from the thigh to the toe
        x_local = r_local_FR[0]
        z_local = r_local_FR[2]
        r_FR = sqrt(x_local**2 + z_local**2)
        cos_theta2 = (r_FR**2 - self.l_thigh**2 - self.l_calf**2)/(2*self.l_thigh*self.l_calf)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2_FR = -acos(cos_theta2) # is always negative
        alpha = atan2(x_local, z_local)
        beta = atan2(self.l_calf*abs(sin(theta2_FR)), -(self.l_thigh+self.l_calf*cos(theta2_FR)))
        theta1_FR = alpha - beta

        theta0_RL = 0.0
        toe_pos = pd_leg[6:9] # from the frame of the base
        thigh_pos = self.p_thigh_RL # from the frame of the base
        r_local_RL = toe_pos - thigh_pos # the vector pointing from the thigh to the toe
        x_local = r_local_RL[0]
        z_local = r_local_RL[2]
        r_RL = sqrt(x_local**2 + z_local**2)
        cos_theta2 = (r_RL**2 - self.l_thigh**2 - self.l_calf**2)/(2*self.l_thigh*self.l_calf)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2_RL = -acos(cos_theta2) # is always negative
        alpha = atan2(x_local, z_local)
        beta = atan2(self.l_calf*abs(sin(theta2_RL)), -(self.l_thigh+self.l_calf*cos(theta2_RL)))
        theta1_RL = alpha - beta

        theta0_RR = 0.0
        toe_pos = pd_leg[9:12] # from the frame of the base
        thigh_pos = self.p_thigh_RR # from the frame of the base
        r_local_RR = toe_pos - thigh_pos # the vector pointing from the thigh to the toe
        x_local = r_local_RR[0]
        z_local = r_local_RR[2]
        r_RR = sqrt(x_local**2 + z_local**2)
        cos_theta2 = (r_RR**2 - self.l_thigh**2 - self.l_calf**2)/(2*self.l_thigh*self.l_calf)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2_RR = -acos(cos_theta2) # is always negative
        alpha = atan2(x_local, z_local)
        beta = atan2(self.l_calf*abs(sin(theta2_RR)), -(self.l_thigh+self.l_calf*cos(theta2_RR)))
        theta1_RR = alpha - beta


        self.qd = np.array([theta0_FL, theta1_FL, theta2_FL,
                    theta0_FR, theta1_FR, theta2_FR,
                    theta0_RL, theta1_RL, theta2_RL,
                    theta0_RR, theta1_RR, theta2_RR])

        if self.num_cycles == 4:
            print(f'self.x_list: {self.x_list}, self.z_list: {self.z_list}')

        return self.qd
    
        
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
    # cam.azimuth = 150
    cam.azimuth = 90
    cam.elevation = 0

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

    x_vals = []
    z_vals = []
    num_iter = 0

    # main simulation loop
    while (not glfw.window_should_close(window)) and (t_sim < max_sim_time):

        t0_sim = data.time

        num_iter += 1

        while (t_sim - t0_sim) < (dt_render):

            t1_wall = time.time()
            t_sim = data.time

            if counter % decimation == 0:

                # generslized position
                q = data.qpos

                # generslized velocity
                v = data.qvel



                q_joints = q[mj_idx.q_joint_idx]
                v_joints = v[mj_idx.v_joint_idx]
                

                if num_iter % 20 == 0:
                    p_imp, _, _, _ = traj.chain_fl.fkin([q_joints[0], q_joints[1], q_joints[2]])
                    x_vals.append(p_imp[0])
                    z_vals.append(p_imp[2])

                if traj.num_cycles == 4:
                    print(f'x_vals: {x_vals}, z_vals: {z_vals}')

                # print(f'\ncurrent q_joints: {q_joints}')

                # print(f'\nError is {np.subtract(q_joints, q_joints_des)}')
                # time.sleep(5)
                if t_sim < 0.0:
                    q_joints_des = traj.evaluate(0.0, dt_sim)


                else:
                    # enter the controller block(?) by calling the evaluate function on the object
                    q_joints_des = traj.evaluate(t0_sim, dt_sim)

                # print(f'q_joints_des are {q_joints_des}')
                
                v_joints_des = np.zeros_like(v_joints)

                # compute torque from PID
                u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

                # u = -kp * (q_joints - q_joints_des)

            #

            # set angular velocities to zero
            data.qvel[mj_idx.ANG_X] = 0
            # set the control torque
            data.ctrl[:] = u

            data.qpos[0] = 0 # comment out
            data.qpos[1] = 0
            data.qpos[2] = 1  # comment out
            data.qpos[3] = 1
            data.qpos[4] = 0
            data.qpos[5] = 0
            data.qpos[6] = 0

            data.qvel[0] = 0 # comment out
            data.qvel[1] = 0
            data.qvel[2] = 0 # comment out
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
