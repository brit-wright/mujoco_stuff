#!/usr/bin/env python3.10

import mujoco
import glfw

import numpy as np
import yaml
import time

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


        # define the foot positions
        self.p0 = np.array([0+0.1934+0+0, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0+0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213])


        self.p0L = np.array([0+0.1934+0+0, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0+0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213])

        self.pmidL = np.array([0+0.1934+0+0+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213+0.1,
                             0+0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213+0.1,
                             0-0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213])
        
        self.pfinalL = np.array([0+0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0+0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213-0.213,
                             0-0.1934+0+0, 0-0.0465-0.0955+0, 0.455+0+0-0.213-0.213])
        
        
        self.p0R = self.pfinalL

        self.pmidR = np.array([0+0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213,
                             0+0.1934+0+0+0.4, 0-0.0465-0.0955+0, 0.455+0+0-0.213+0.1,
                             0-0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213,
                             0-0.1934+0+0+0.4, 0-0.0465-0.0955+0, 0.455+0+0-0.213+0.1])

        self.pfinalR = np.array([0+0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213,
                             0+0.1934+0+0+0.4+0.4, 0-0.0465-0.0955+0, 0.455+0+0-0.213,
                             0-0.1934+0+0+0.4+0.4, 0+0.0465+0.0955+0, 0.455+0+0-0.213,
                             0-0.1934+0+0+0.4+0.4, 0-0.0465-0.0955+0, 0.455+0+0-0.213])
        
        self.pd = self.p0

        self.R0 = np.eye(3)

        self.qd = self.q0

        self.Rd = self.R0

        self.cycle = 0

        self.lam = 10

        self.urdf = './models/go2/go2.urdf'

        # set up the kinematic chains for each leg
        self.chain_fl = KinematicChain('base', 'FL_foot', self.jointnames()[0:3], self.urdf)
        self.chain_fr = KinematicChain('base', 'FR_foot', self.jointnames()[3:6], self.urdf)
        self.chain_rl = KinematicChain('base', 'RL_foot', self.jointnames()[6:9], self.urdf)
        self.chain_rr = KinematicChain('base', 'RR_foot', self.jointnames()[9:12], self.urdf)


        # initialize the position of the body
        self.pbody = np.array([0.0, 0.0, 0.355])



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

        # define the step increment positions
        inc_left_mid = np.array([0.4, 0.0, 0.1,
                                 0.0, 0.0, 0.0,
                                 0.4, 0.0, 0.1,
                                 0.0, 0.0, 0.0])
        
        inc_left_final = np.array([0.4, 0.0, -0.1,
                                   0.0, 0.0, 0.0,
                                   0.4, 0.0, -0.1,
                                   0.0, 0.0, 0.0])
        
        inc_right_mid = np.array([0.0, 0.0, 0.0,
                                  0.4, 0.0, 0.1,
                                  0.0, 0.0, 0.0,
                                  0.4, 0.0, 0.1])
        
        inc_right_final = np.array([0.0, 0.0, 0.0,
                                    0.4, 0.0, -0.1,
                                    0.0, 0.0, 0.0,
                                    0.4, 0.0, -0.1])
        
    
        # take the current time step and use this to determine what step in the
        # cycle we're at

        if t - self.cycle * 4 > 4:

            self.p0L = self.pfinalR

            self.pmidL = self.p0L + inc_left_mid

            self.pfinalL = self.pmidL + inc_left_final

            self.p0R = self.pfinalL

            self.pmidR = self.p0R + inc_right_mid

            self.pfinalR = self.pmidR + inc_right_final

            self.lam += 5

            self.cycle += 1

        # calculate the new time 
        t = t % 4.0

        if t == 0.0:

            pd_leg = self.p0
            vd_leg = np.zeros_like(pd_leg)


        # calculate the desired leg (toe) positions and velocities
        if t < 1.0 and t > 0.0:

            (s0, s0dot) = self.goto(t, 1.0, 0.0, 1.0)

            pd_leg = self.p0L + (self.pmidL - self.p0L) * s0
            vd_leg = (self.pmidL - self.p0L) * s0dot

        elif t < 2.0 and t > 1.0:

            (s0, s0dot) = self.goto(t-1.0, 1.0, 0.0, 1.0)

            pd_leg = self.pmidL + (self.pfinalL - self.pmidL) * s0
            vd_leg = (self.pfinalL - self.pmidL) * s0dot

        elif t < 3.0 and t > 2.0:

            (s0, s0dot) = self.goto(t-2.0, 1.0, 0.0, 1.0)

            pd_leg = self.p0R + (self.pmidR - self.p0R) * s0
            vd_leg = (self.pmidR - self.p0R) * s0dot

        elif t < 4.0 and t > 3.0:

            (s0, s0dot) = self.goto(t-3.0, 1.0, 0.0, 1.0)

            pd_leg = self.pmidR + (self.pfinalR - self.pmidR) * s0
            vd_leg = (self.pfinalR - self.pmidR) * s0dot

        elif t == 4.0:

            pd_leg = self.p0
            vd_leg = np.zeros_like(pd_leg)

        
        # Storing the previous desired joint and position values

        qdlast = self.qd
        pbody_last = self.pbody
        pdlast = self.pd
        Rdlast = self.Rd

        # Compute the old forward kinematics for each leg using the kinematic chains
        # print(f'self_chain_fl is {self.chain_fl}')
        # print(f'qdlast is {qdlast}')
        # print(f'self.fl is {self.fl}')


        (p_leg1, R_leg1, Jv_leg1, Jw_leg1) = self.chain_fl.fkin(qdlast[self.fl[0]: self.fl[-1]+1])
        (p_leg2, R_leg2, Jv_leg2, Jw_leg2) = self.chain_fr.fkin(qdlast[self.fr[0]: self.fr[-1]+1])
        (p_leg3, R_leg3, Jv_leg3, Jw_leg3) = self.chain_rl.fkin(qdlast[self.rl[0]: self.rl[-1]+1])
        (p_leg4, R_leg4, Jv_leg4, Jw_leg4) = self.chain_rr.fkin(qdlast[self.rr[0]: self.rr[-1]+1])

        # matrix to fill unused positions in the Jacobian (for the leg and body)
        filler = np.zeros((3,3))

        # identity matrix for maintaining linear velocity of body in the Jacobian
        unity = np.eye(3)

        # Refer to Gunter's Multilimb document. I'll try to explain as best as I can
        # currently re-living me133a trauma :)

        # all the desired leg positions for this current stage in the gait
        pd_FL = pd_leg[0:3]
        pd_FR = pd_leg[3:6]
        pd_RL = pd_leg[6:9]
        pd_RR = pd_leg[9:12]

        # the positions the legs ended in from the previous tage in the gait
        p_FL = pbody_last + Rdlast @ p_leg1
        p_FR = pbody_last + Rdlast @ p_leg2
        p_RL = pbody_last + Rdlast @ p_leg3
        p_RR = pbody_last + Rdlast @ p_leg4

        # calculate the new linear Jacobian based on the Jacobian from the previous step
        jacob_FL = Rdlast @ Jv_leg1
        jacob_FR = Rdlast @ Jv_leg2
        jacob_RL = Rdlast @ Jv_leg3
        jacob_RR = Rdlast @ Jv_leg4

        # calculate the new angular Jacobian based on the Jacobian from the previous step
        jacob_FLw = -crossmat(Rdlast @ pd_FL)
        jacob_FRw = -crossmat(Rdlast @ pd_FR)
        jacob_RLw = -crossmat(Rdlast @ pd_RL)
        jacob_RRw = -crossmat(Rdlast @ pd_RR)

        # construct the full Jacobian matrix by stacking the individual rows
        Jrow1 = np.hstack((jacob_FL, filler, filler, filler, unity, jacob_FLw))
        Jrow2 = np.hstack((filler, jacob_FR, filler, filler, unity, jacob_FRw))
        Jrow3 = np.hstack((filler, filler, jacob_RL, filler, unity, jacob_RLw))
        Jrow4 = np.hstack((filler, filler, filler, jacob_RR, unity, jacob_RRw))

        J_big = np.vstack((Jrow1, Jrow2, Jrow3, Jrow4))

        # combine the positions into a giant column vector
        p_leg_combined = np.concatenate((p_FL, p_FR, p_RL, p_RR))

        # TODO: check if I can replace this with pd_leg
        pd_leg_combined = np.concatenate((pd_FL, pd_FR, pd_RL, pd_RR))

        # TODO: check if this can be replaced with something else
        for i in range(0, 9):
            vd_leg[i] = 0.0

        # compute the velocity based on the position error between the previous desired
        # position and the resulting foot/toe positions in the last iteration
        xrdot = vd_leg + self.lam * ep(pdlast, p_leg_combined)

        print(f'xrdot is {xrdot}')

        # we also want to have a secondary task to steer the leg-joints away from singlularities
        q_goal = 1.5 * np.array([0.0, 0.45, -0.9,
                                0.0, 0.45, -0.9,
                                0.0, 0.45, -0.9,
                                0.0, 0.45, -0.9,
                                0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0])

        lam_secondary = 2
        q_secondary = lam_secondary * np.subtract(q_goal, self.qd)
        # qd_dot = np.add(np.linalg.pinv(J_big) @ xrdot, np.subtract(np.identity(18), np.linalg.pinv(J_big) @ J_big) @ q_secondary)
        qd_dot = (np.linalg.pinv(J_big) @ xrdot)
        qd = qdlast + dt * qd_dot

        # print(f'qd_dot is {qd_dot}')
        # print(f'dt is {dt}')
        # print(f'qdlast is {qdlast}')
        # print(f'qd is {qd}')
        
        # Update the position and oriantation of the body
        self.pbody = pbody_last + dt * qd_dot[12:15] # who is this?
        Rbody = Rdlast + dt * crossmat(qd_dot[15:]) @ Rdlast # korewa nan desuka?

        # Normalize the orientation to maintain orthonormality
        u, s, vt = np.linalg.svd(Rbody)
        Rbody = u @ vt

        # Save the desired joint and foot (tip) positions for the next cycle
        self.qd = qd
        self.pd = pd_leg_combined

        # Create the transform matrix for the body
        Tbody = T_from_Rp(Rbody, self.pbody)

        # return the joint values for the legs

        qd_dot2 = np.zeros_like(qd_dot[0:12])

        return(qd[0:12], qd_dot2)
    
        

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

                if t_sim <= 2.0:
                    q_joints_des = np.zeros_like(q_joints)
                    v_joints_des = np.zeros_like(v_joints)

                else:

                    # enter the controller block(?) by calling the evaluate function on the object
                    q_joints_des, v_joints_des = traj.evaluate(t0_sim, dt_sim)

                    # print(f'q_joints_des are {q_joints_des}')
                    

                # compute torque from PID
                # u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

                u = -2 * (q_joints - q_joints_des)


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
