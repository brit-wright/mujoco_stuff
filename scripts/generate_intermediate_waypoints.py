import numpy as np
from math import sqrt, ceil

def generate_waypoints(points):

    max_dist = 5.0
    
    way_points_list = []

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

        for pt in way_points: way_points_list.append(pt)

    way_points_list.append(points[i+1])

    return way_points_list


points = [[0.0, 0.0, 0.1], [21, 6, 0.1], [34, 8, 0.1], [36, 8, 0.1]]
inters = generate_waypoints(points)

print(f'inters: {inters}')