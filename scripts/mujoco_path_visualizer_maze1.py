import mujoco
from mujoco.viewer import launch
import solve_maze_using_RRT_maze1 as solveMaze
from math import sqrt, ceil
import csv
from datetime import datetime
# Features
# This script basically takes in the waypoints and updates the .xml file
# to include them in the Mujoco model

def generate_waypoints(points):

    max_dist = 3.0
    
    way_points_list = []

    intermediates = []

    for i in range(len(points)-1):


        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[i+1][0], points[i+1][1]

        way_points = [points[i]]
        start_ind = i

        # check the distance between the points
        distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)

        if distance > max_dist:

            # calculate number of waypoints
            num_waypoints = ceil(distance/max_dist)

            for idx in range(num_waypoints-1):
                idx+=1
                way_x = x1 + (x2 - x1) * idx/num_waypoints
                way_y = y1 + (y2 - y1) * idx/num_waypoints

                way_points.insert(start_ind+idx, [way_x, way_y, 0.1])

                intermediates.append([way_x, way_y, 0.1])

        for pt in way_points: way_points_list.append(pt)

    way_points_list.append(points[i+1])

    return way_points_list, intermediates

def add_path_to_xml(path_nodes, intermediates):
    
    geoms = []

    # start by adding the nodes to the .xml file as spheres. the .xml syntax looks like this 
    # <geom name="node1" type="sphere" size="0.1" pos="x y z" rgba="0 0 1 1"/>

    for index, (x, y, z) in enumerate(path_nodes):

        if [x, y, z] in intermediates:
            geom = f'<geom name="node{index+1}" type="sphere" size="0.1" pos="{x} {y} {z}" rgba="1 1 0 0.5" contype="2" conaffinity="2"/>'
            geoms.append(geom)
        else:
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

def main():

    path_nodes0 = []
    path_nodes_path = '/home/brittany/IRIS_env_drake/maze1_results.csv'
    with open(path_nodes_path) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            path_nodes0.append([float(row[0]), float(row[1]), 0.1])

    # filter duplicate nodes
    path_nodes = []
    for node in path_nodes0:
        if node not in path_nodes:
            path_nodes.append(node)

    print(f'path_nodes: {path_nodes}')

    inters, intermediates = generate_waypoints(path_nodes)

    print(f'waypoints: {inters}')

    path_geoms = add_path_to_xml(inters, intermediates)

    # base_xml_path = './models/go2/noel_maze_v0.xml'
    base_xml_path = './models/go2/second_maze.xml'

    tnow = datetime.now()
    new_xml_path = './models/go2/maze1_with_geoms'+'_'+str(tnow.month)+'_'+str(tnow.day)+'_'+str(tnow.year)+'_'+str(tnow.hour)+'_'+str(tnow.minute)+'.xml'

    phrase = new_xml_path[30:]

    with open(base_xml_path, "r") as f:
        base_xml = f.read()

    new_xml = base_xml.replace("</worldbody>", path_geoms + "\n</worldbody>")

    with open(new_xml_path, "w") as f:
        f.write(new_xml)
        
    print(f'intermediates: {intermediates}')

    model = mujoco.MjModel.from_xml_path(new_xml_path)
    launch(model)

    return new_xml_path, inters, intermediates, phrase

if __name__ == "__main__":
    main()




