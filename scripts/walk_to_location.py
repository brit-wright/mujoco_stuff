# Features
# WALKING FORWARD
# Starts from some user-defined stable configuration
# Uses a double-stance walking gait
# Newton-Raphson Method for Inverse Kinematics
# Calculates z-foot position using Bezier curves
# Using Raibert's method to get rid of y-drift

#!/usr/bin/env python3.10

import mujoco
import glfw

import numpy as np
import yaml
import time
from math import acos, atan2, sqrt, sin, cos

from definitions.go2_definitions import Mujoco_IDX_go2
from QuadrupedControllerv1 import QuadrupedController
from hw5code.TransformHelpers import *


def quaternion_to_yaw(w, x, y, z):
    return atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# main function to run the system
if __name__ == '__main__':

    # load the config file
    config_file = './config/go2_config.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # load the mujoco model
    mj_model_path = "./models/go2/scene.xml"
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
    cam.distance = 3.0
    # cam.elevation = -1
    cam.elevation = -35
    cam.azimuth = 270

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
    robot = QuadrupedController(data, mj_idx, q0_joint)

    q = data.qpos
    v = data.qvel

    q_joints = q[mj_idx.q_joint_idx]
    v_joints = v[mj_idx.v_joint_idx]

    q_joints_des = np.zeros_like(q_joints)

    T_stand = 0.5
    T_stab = 1
    T_run = 5
    T_reset = 1
    deltx = 0.0
    z_command = 0.35
    v_bod = 0.6

    goal_pos = [5, 15]

    k_mag = 1.0
    k_theta = 1.0
    k_x = 1.0
    k_y = 1.0

    # forward_max = 0.5
    # turn_max = 0.8

    forward_max = 0.5
    turn_max = 0.5

    gait_start = True

    # main simulation loop
    while (not glfw.window_should_close(window)) and (t_sim < max_sim_time):

        t0_sim = data.time

        # t_sim is the total/running simulation
        while (t_sim - t0_sim) < (dt_render):

            # print(f'Current x position is: {data.qpos[mj_idx.POS_X]}')

            t1_wall = time.time()
            t_sim = data.time

            if counter % decimation == 0:

                # generalized position
                q = data.qpos

                # generslized velocity
                v = data.qvel

                q_joints = q[mj_idx.q_joint_idx]
                v_joints = v[mj_idx.v_joint_idx]
                v_joints_des = np.zeros_like(v_joints)

                p_curr = q[mj_idx.q_base_pos_idx]


                theta_curr = quaternion_to_yaw(q[mj_idx.QUAT_W], q[mj_idx.QUAT_X], q[mj_idx.QUAT_Y], q[mj_idx.QUAT_Z])
                theta_desired = atan2(goal_pos[1] - p_curr[1], goal_pos[0] - p_curr[0])
                angle_error = normalize_angle(theta_desired - theta_curr)
                wz_command = k_theta * angle_error

                x_error = goal_pos[0] - p_curr[0]
                vx_command = k_x * x_error

                y_error = goal_pos[1] - p_curr[1]
                vy_command = k_y * y_error

                # clipping the values
                if wz_command >= 0.0:
                    wz_command = np.clip(wz_command, 0.1, turn_max)
                elif wz_command < 0.0:
                    wz_command = np.clip(wz_command, -0.1, -turn_max)


                vx_command = np.clip(vx_command, 0.2, forward_max)
                vy_command = np.clip(vy_command, 0.2, forward_max)

                commands = np.array([vx_command, 0.0, wz_command])

                errors = np.array([x_error, y_error, angle_error])

                if t_sim <= T_stand:
                    # print('Standing')
                    q_joints_des = robot.stand()
                
                else:
                    t_curr = t_sim - T_stand
                    q_joints_des = robot.walker(t_curr, commands, errors, theta_curr, gait_start)
                

                # compute torque from PID
                u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

            # set the control torque
            data.ctrl[:] = u


            # data.qpos[0] = 0 # comment out
            # data.qpos[1] = 0
            # data.qpos[2] = 1  # comment out
            # data.qpos[3] = 1
            # data.qpos[4] = 0
            # data.qpos[5] = 0
            # data.qpos[6] = 0

            # data.qvel[0] = 0 # comment out
            # data.qvel[1] = 0
            # data.qvel[2] = 0 # comment out
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