import numpy as np
import mujoco
from mujoco.viewer import launch
import os
# Features
# This script basically takes in the waypoints and updates the .xml file
# to include them in the Mujoco model

def add_path_to_xml(path_nodes):
    
    geoms = []

    # start by adding the nodes to the .xml file as spheres. the .xml syntax looks like this 
    # <geom name="node1" type="sphere" size="0.1" pos="x y z" rgba="0 0 1 1"/>

    for index, (x, y, z) in enumerate(path_nodes):

        geom = f'<geom name="node{index+1}" type="sphere" size="0.1" pos="{x} {y} {z}" rgba="0 0 1 0.5" contype="2" conaffinity="2"/>'
        geoms.append(geom)

    # add the capsules to the .xml file to connect the nodes to one another. syntax looks like this
    # <geom name="cap1to2" type="capsule" fromto="21 7 0.1  21 9 0.1" size="0.1" rgba="0 1 0 0.5" contype="2" conaffinity="2"/>
    
    for index in range(len(path_nodes) - 1):

        x1, y1, z1 = path_nodes[index]
        x2, y2, z2 = path_nodes[index + 1]

        geom = f'<geom name="cap{index+1}to{index+2}" type="capsule" fromto="{x1} {y1} {z1}  {x2} {y2} {z2}" size="0.1" rgba="0 1 0 0.5" contype="2" conaffinity="2"/>'
        geoms.append(geom)

    return "\n".join(geoms)

path_nodes = np.array([[21, 6, 0.1],
                       [21, 8, 0.1],
                       [21, 10, 0.1]])

path_geoms = add_path_to_xml(path_nodes)

base_xml_path = './models/go2/noel_maze.xml'
new_xml_path = './models/go2/maze_with_geoms.xml'

with open(base_xml_path, "r") as f:
    base_xml = f.read()

new_xml = base_xml.replace("</worldbody>", path_geoms + "\n</worldbody>")

with open(new_xml_path, "w") as f:
    f.write(new_xml)
    
# model = mujoco.MjModel.from_xml_path(new_xml_path)
# launch(model)



