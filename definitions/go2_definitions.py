# clas to hold the indices of the model instance
class Mujoco_IDX_go2():

    # generalized positions

    # defining the base positions
    POS_X = 0
    POS_Y = 1
    POS_Z = 2
    QUAT_W = 3
    QUAT_X = 4
    QUAT_Y = 5
    QUAT_Z = 6

    # defining the leg joint positions
    POS_FLH = 7
    POS_FLT = 8
    POS_FLC = 9
    POS_FRH = 10
    POS_FRT = 11
    POS_FRC = 12
    POS_RLH = 13
    POS_RLT = 14
    POS_RLC = 15
    POS_RRH = 16
    POS_RRT = 17
    POS_RRC = 18

    
    # generalized velocities

    # defining the base velocities
    VEL_X = 0
    VEL_Y = 1
    VEL_Z = 2
    ANG_X = 3
    ANG_Y = 4
    ANG_Z = 5

    # defining the velocities of the leg joints
    VEL_FLH = 6
    VEL_FLT = 7
    VEL_FLC = 8
    VEL_FRH = 9
    VEL_FRT = 10
    VEL_FRC = 11
    VEL_RLH = 12
    VEL_RLT = 13
    VEL_RLC = 14
    VEL_RRH = 15
    VEL_RRT = 16
    VEL_RRC = 17

    # base index
    q_base_pos_idx = [POS_X, POS_Y, POS_Z]
    q_base_quat_idx = [QUAT_W, QUAT_X, QUAT_Y, QUAT_Z]
    v_base_vel_idx = [VEL_X, VEL_Y, VEL_Z]
    v_base_ang_idx = [ANG_X, ANG_Y, ANG_Z]

    # joints
    q_joint_idx = [POS_FLH, POS_FLT, POS_FLC,
                   POS_FRH, POS_FRT, POS_FRC,
                   POS_RLH, POS_RLT, POS_RLC,
                   POS_RRH, POS_RRT, POS_RRC]
    
    v_joint_idx = [VEL_FLH, VEL_FLT, VEL_FLC,
                   VEL_FRH, VEL_FRT, VEL_FRC,
                   VEL_RLH, VEL_RLT, VEL_RLC,
                   VEL_RRH, VEL_RRT, VEL_RRC]
    
    
    
    