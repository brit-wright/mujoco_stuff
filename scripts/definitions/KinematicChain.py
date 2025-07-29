'''
KinematicChain.py

Updated Kinematic Chain Code for Problem 3.

This program processes the URDF to form a kinematic chain and computes
the position, orientation, and Jacobian for a given set of joint values.
'''

import enum
import rclpy
import numpy as np

from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from urdf_parser_py.urdf import Robot

# Import transformation helpers
from hw5code.TransformHelpers import *

#
#   Single URDF Step
#
#   This captures a single step from one frame to the next.  It be of type:
#
#     FIXED     Just a fixed T-matrix shift, nothing moving, not a DOF.
#     REVOLUTE  A fixed T-matrix shift, followed by a rotation about an axis.
#     LINEAR    A fixed T-matrix shift, followed by a transation along an axis.
#
#   It contains several pieces of permanent data (coming from the URDF):
#
#     name      String showing the name
#     type      One of the above
#     Tshift    Fixed shift: Transform of this frame w.r.t. previous
#     nlocal    Joint axis (if applicable) in the local frame
#
#   We also add information how this relates to the active joints:
#
#     dof       If an active dof (not FIXED), the dof number
#

# Define the joint types.
class JointType(enum.Enum):
    FIXED = 0
    REVOLUTE = 1
    LINEAR = 2


# Define a single step in the URDF (kinematic chain).
class URDFStep():
    def __init__(self, name, type, Tshift, nlocal):
        # Store the permanent/fixed/URDF data.
        self.name = name  # Joint name
        self.type = type  # Joint type (per above enumeration)
        self.Tshift = Tshift  # Transform w.r.t. previous frame
        self.nlocal = nlocal  # Joint axis in the local frame

        # Match against the joint numbers
        self.dof = None  # Joint DOF number (or None if FIXED)


#
#   Kinematic Chain Object
#
#   This stores the information provided by the URDF in the form of
#   steps (see above).  In particular, see the fkin() function, as it
#   walks up the chain to determine the transforms.
#

# Define the full kinematic chain
class KinematicChain():
    # Helper functions for printing info and errors.
    def info(self, string):
        # self.node.get_logger().info("KinematicChain: " + string)
        print(string)

    def error(self, string):
        # self.node.get_logger().error("KinematicChain: " + string)
        raise Exception(string)

    # Initialization - load the URDF!
    def __init__(self, baseframe, tipframe, expectedjointnames, urdf_name):

        # Prepare the list the steps.
        self.steps = []
        self.dofs = 0

        try:
            with open(urdf_name, 'rb') as file:
                self.urdf = file.read()

                # print(f'Self.URDF is {self.urdf}')
        except FileNotFoundError:
            print("Error: URDF not found")
        
        # Convert the URDF string into a Robot object and report.
        robot = Robot.from_xml_string(self.urdf)
        self.info(f"Processing URDF for robot '{robot.name}'")

        # Parse the Robot object into a list of URDF steps from the
        # base frame to the tip frame.  Search backwards, as the robot
        # could be a tree structure: while a parent may have multiple
        # children, every child has only one parent.  The resulting
        # chain of steps is unique.
        frame = tipframe  # in the example, frame = 'tip' and baseframe = 'world'
        while frame != baseframe:
            # Look for the URDF joint to the parent frame.
            joint = next((j for j in robot.joints if j.child == frame), None)
            if joint is None:
                self.error(f"Unable to find joint connecting to '{frame}'")
            if joint.parent == frame:
                self.error(f"Joint '{joint.name}' connects '{frame}' to itself")
            frame = joint.parent

            # Check the type (use the above enumeration)
            if joint.type in ['revolute', 'continuous']:
                type = JointType.REVOLUTE
            elif joint.type == 'prismatic':
                type = JointType.LINEAR
            elif joint.type == 'fixed':
                type = JointType.FIXED
            else:
                self.error(f"Joint '{joint.name}' has unknown type '{joint.type}'")

            # Convert the URDF information into a single URDF step.
            # Note the axis (nlocal) is meaningless for a fixed joint.
            self.steps.insert(0, URDFStep(
                name=joint.name,
                type=type,
                Tshift=T_from_URDF_origin(joint.origin),
                nlocal=n_from_URDF_axis(joint.axis)))

        # Set the active DOF numbers walking up the steps.
        dof = 0
        for step in self.steps:
            if step.type != JointType.FIXED:
                step.dof = dof
                dof += 1
            else:
                step.dof = None
        self.dofs = dof
        # self.info(f"URDF has {len(self.steps)} steps, {self.dofs} active DOFs:")

        # Report what we found.
        for (i, step) in enumerate(self.steps):
            string = f"Step #{i} {step.type.name:<8}"
            string += "      " if step.dof is None else f"DOF #{step.dof}"
            string += f" '{step.name}'"
            self.info(string)

        # Confirm the active joint names matches the expectation
        jointnames = [s.name for s in self.steps if s.type != JointType.FIXED]
        if jointnames != list(expectedjointnames):
            self.error("Chain does not match the expected names: " +
                       str(expectedjointnames))

    # Compute the forward kinematics!
    def fkin(self, q):
        # Check the number of joints
        if len(q) != self.dofs:
            self.error("Number of joint angles (%d) does not chain (%d)" %
                       (len(q), self.dofs))
            
        # We will build up three lists.  For each non-fixed (real)
        # joint remember the type, position, axis w.r.t. world.
        type = []
        p = []
        n = []

        # Initialize the T matrix to walk up the chain, w.r.t. world frame!
        T = np.eye(4)

        # Walk the chain, one URDF step at a time, adjusting T as we
        # go.  This could be a fixed or active joint.
        for step in self.steps:
            # Update the transform with the fixed shift from the URDF.
            T = T @ step.Tshift

            # For revolute and linear joints, account for the joint value
            if step.type == JointType.REVOLUTE:
                # Revolute joint: apply rotation about the joint axis
                T = T @ T_from_Rp(Rotn(step.nlocal, q[step.dof]), np.zeros(3))
            elif step.type == JointType.LINEAR:
                # Linear joint: apply translation along the joint axis
                T = T @ T_from_Rp(np.eye(3), step.nlocal * q[step.dof])

            # For active joints, store the type, position, and axis info
            if step.type != JointType.FIXED:
                type.append(step.type)
                p.append(p_from_T(T))
                n.append(R_from_T(T) @ step.nlocal)

        # Collect the tip information.
        ptip = p_from_T(T)
        Rtip = R_from_T(T)

        # Collect the Jacobian for each active joint.
        Jv = np.zeros((3, self.dofs))
        Jw = np.zeros((3, self.dofs))
        for i in range(self.dofs):
            # Fill in the appropriate Jacobian column based on the type.
            if type[i] == JointType.REVOLUTE:
                # Revolute joint: rotational contribution to Jv and Jw
                Jv[:, i] = cross(n[i], ptip - p[i])
                Jw[:, i] = n[i]
            elif type[i] == JointType.LINEAR:
                # Linear joint: translational contribution to Jv
                Jv[:, i] = n[i]
                Jw[:, i] = np.zeros(3)

        # Return the info
        return (ptip, Rtip, Jv, Jw)


#
#   Main Code
#
#   This simply tests the kinematic chain and associated calculations!
#
def main(args=None):
    # Set the print options to something we can read.
    np.set_printoptions(precision=3, suppress=True)

    # Initialize ROS and the node.
    rclpy.init(args=args)

    # Set up the kinematic chain object, assuming the 3 DOF.
    jointnames = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint']
    baseframe = 'base'
    tipframe = 'FR_foot'
    urdf_name = './models/go2/go2.urdf'
    chain = KinematicChain(baseframe, tipframe, jointnames, urdf_name)

    # Define the test.
    def test(q):
        (ptip, Rtip, Jv, Jw) = chain.fkin(q)
        print('q:\n', np.degrees(q))
        print('ptip(q):\n', ptip)
        print('Rtip(q):\n', Rtip)
        print('Jv(q):\n', Jv)
        print('Jw(q):\n', Jw)
        print('----------------------------------------')

    # Run the tests.
    test(np.radians(np.array([20.0, 40.0, -30.0])))
    test(np.radians(np.array([30.0, 30.0, 60.0])))
    test(np.radians(np.array([-45.0, 75.0, 120.0])))


if __name__ == "__main__":
    main()
