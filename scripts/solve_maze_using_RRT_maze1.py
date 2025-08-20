import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
import csv
# PARAMETERS
STEP_SIZE = 0.5 # chose a static step size
SMAX = 20000  # maximum number of step attempts
NMAX = 10000 # maximum number of nodes

# Define the list of obstacles/objects as well as star/goal

# rect1 = LineString([[0.8, 9.2],
#                     [3.2, 9.2],
#                     [3.2, 8],
#                     [0.8, 8],
#                     [0.8, 9.2]])

# rect2 = LineString([[0.8, 8],
#                     [2.2, 8],
#                     [2.2, 5.8],
#                     [0.8, 5.8],
#                     [0.8, 8]])

# rect3 = LineString([[2.8, 7.2], 
#                     [4.2, 7.2],
#                     [4.2, 5],
#                     [2.8, 5],
#                     [2.8, 7.2]])

# rect4 = LineString([[1.8, 5],
#                     [1.8, 3.8],
#                     [4.2, 3.8],
#                     [4.2, 5],
#                     [1.8, 5]])

# rect5 = LineString([[0.8, 3.2],
#                     [0.8, 1.8],
#                     [5.2, 1.8],
#                     [5.2, 3.2],
#                     [0.8, 3.2]])

# rect6 = LineString([[2.8, 0],
#                     [2.8, 1.2],
#                     [6, 1.2],
#                     [6, 0],
#                     [2.8, 0]])

# rect7 = LineString([[6, 2.2],
#                     [7.2, 2.2],
#                     [7.2, 0],
#                     [6, 0],
#                     [6, 2.2]])

# rect8 = LineString([[6.8, 7.2],
#                     [8.2, 7.2],
#                     [8.2, 2.8],
#                     [6.8, 2.8],
#                     [6.8, 7.2]])

# rect9 = LineString([[7.8, 0.8],
#                     [7.8, 2.2],
#                     [9, 2.2],
#                     [9, 0.8],
#                     [7.8, 0.8]])

# rect10 = LineString([[9, 0],
#                      [9, 3.2],
#                      [10, 3.2],
#                      [10, 0],
#                      [9, 0]])

# rect11 = LineString([[4.8, 3.8],
#                      [4.8, 9.2],
#                      [6, 9.2],
#                      [6, 3.8],
#                      [4.8, 3.8]])

# rect12 = LineString([[6, 8],
#                      [6, 9.2],
#                      [10, 9.2],
#                      [10, 8],
#                      [6, 8]])

# rect13 = LineString([[8.8, 3.8],
#                      [10, 3.8],
#                      [10, 8],
#                      [8.8, 8],
#                      [8.8, 3.8]])

# rect14 = LineString([[2.2, 8],
#                      [2.2, 7.8],
#                      [3.2, 7.8],
#                      [3.2, 8],
#                      [2.2, 8]])

# rect15 = LineString([[1.8, 5],
#                      [1.8, 5.2],
#                      [2.8, 5.2],
#                      [2.8, 5],
#                      [1.8, 5]])

# rect16 = LineString([[5.8, 2.2],
#                      [5.8, 1.2],
#                      [6, 1.2],
#                      [6, 2.2],
#                      [5.8, 2.2]])

# rect17 = LineString([[6, 3.8],
#                      [6, 8],
#                      [6.2, 8],
#                      [6.2, 3.8],
#                      [6, 3.8]])

# rect18 = LineString([[6.2, 8],
#                      [6.2, 7.8],
#                      [8.8, 7.8],
#                      [8.8, 8],
#                      [6.2, 8]])

# rect19 = LineString([[8.8, 2.2],
#                      [8.8, 3.2],
#                      [9, 3.2],
#                      [9, 2.2],
#                      [8.8, 2.2]])

# rect20 = LineString([[8.8, 0],
#                      [9, 0],
#                      [9, 0.8],
#                      [8.8, 0.8],
#                      [8.8, 0]])

rect1 = LineString([[0.7, 9.3],
                    [3.3, 9.3],
                    [3.3, 8],
                    [0.7, 8],
                    [0.7, 9.3]])

rect2 = LineString([[0.7, 8],
                    [2.3, 8],
                    [2.3, 5.7],
                    [0.7, 5.7],
                    [0.7, 8]])

rect3 = LineString([[2.7, 7.3], 
                    [4.3, 7.3],
                    [4.3, 5],
                    [2.7, 5],
                    [2.7, 7.3]])

rect4 = LineString([[1.7, 5],
                    [1.7, 3.7],
                    [4.3, 3.7],
                    [4.3, 5],
                    [1.7, 5]])

rect5 = LineString([[0.7, 3.3],
                    [0.7, 1.7],
                    [5.3, 1.7],
                    [5.3, 3.3],
                    [0.7, 3.3]])

rect6 = LineString([[2.7, 0],
                    [2.7, 1.3],
                    [6, 1.3],
                    [6, 0],
                    [2.7, 0]])

rect7 = LineString([[6, 2.3],
                    [7.3, 2.3],
                    [7.3, 0],
                    [6, 0],
                    [6, 2.3]])

rect8 = LineString([[6.7, 7.3],
                    [8.3, 7.3],
                    [8.3, 2.7],
                    [6.7, 2.7],
                    [6.7, 7.3]])

rect9 = LineString([[7.7, 0.7],
                    [7.7, 2.3],
                    [9, 2.3],
                    [9, 0.7],
                    [7.7, 0.7]])

rect10 = LineString([[9, 0],
                     [9, 3.3],
                     [10, 3.3],
                     [10, 0],
                     [9, 0]])

rect11 = LineString([[4.7, 3.7],
                     [4.7, 9.3],
                     [6, 9.3],
                     [6, 3.7],
                     [4.7, 3.7]])

rect12 = LineString([[6, 8],
                     [6, 9.3],
                     [10, 9.3],
                     [10, 8],
                     [6, 8]])

rect13 = LineString([[8.7, 3.7],
                     [10, 3.7],
                     [10, 8],
                     [8.7, 8],
                     [8.7, 3.7]])

rect14 = LineString([[2.3, 8],
                     [2.3, 7.7],
                     [3.3, 7.7],
                     [3.3, 8],
                     [2.3, 8]])

rect15 = LineString([[1.7, 5],
                     [1.7, 5.3],
                     [2.7, 5.3],
                     [2.7, 5],
                     [1.7, 5]])

rect16 = LineString([[5.7, 2.3],
                     [5.7, 1.3],
                     [6, 1.3],
                     [6, 2.3],
                     [5.7, 2.3]])

rect17 = LineString([[6, 3.7],
                     [6, 8],
                     [6.3, 8],
                     [6.3, 3.7],
                     [6, 3.7]])

rect18 = LineString([[6.3, 8],
                     [6.3, 7.7],
                     [8.7, 7.7],
                     [8.7, 8],
                     [6.3, 8]])

rect19 = LineString([[8.7, 2.3],
                     [8.7, 3.3],
                     [9, 3.3],
                     [9, 2.3],
                     [8.7, 2.3]])

rect20 = LineString([[8.7, 0],
                     [9, 0],
                     [9, 0.7],
                     [8.7, 0.7],
                     [8.7, 0]])

(xmin, xmax) = (0, 10)
(ymin, ymax) = (0, 10)

(xA, xB, xC, xD) = ( 2, 4, 6, 8)
(yA, yB, yC, yD) = ( 2, 4, 6, 8)

xlabels = (xmin, 1, 2, 3, 4, 5, 6, 7, 8, 9, xmax)
ylabels = (ymin, 1, 2, 3, 4, 5, 6, 7, 8, 9, ymax)


# Draw the outer boundary/walls
outside = LineString([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                      [xmin, ymax], [xmin, ymin]])



walls = prep(MultiLineString([outside, rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, rect9, rect10, rect11, rect12, rect13, rect14, rect15, rect16, rect17, rect18, rect19, rect20]))
bonus   = LineString([[xD, yC], [xD, yC]])


# Define the start/goal states (x, y, theta) of the mattress
# (xstart, ystart) = (21, 6)
# (xgoal, ygoal) = (2, 16)

(xstart, ystart) = (0.6, 0.6)
(xgoal, ygoal) = (9, 9.8)

# Visualization Utility
class Visualization:
    def __init__(self):

        # Clear the current, or create a new figure
        plt.clf()

        # Create new axes, enable the grid, and set the axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_xticks(xlabels)
        plt.gca().set_yticks(ylabels)
        plt.gca().set_aspect('equal')

        # Show the walls
        for l in walls.context.geoms:
            plt.plot(*l.xy, 'k', linewidth=2)
        if bonus in walls.context.geoms:
            plt.plot(*bonus.xy, 'b:', linewidth=3)

        # Show
        self.show()

    def show(self, text = ''):
        # Show the plot
        plt.pause(0.001)

        # If text is specified, print and wait for confirmation
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x), (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

# NODE DEFINITION

class Node:
    def __init__(self, x, y):
        # Define a parent 
        self.parent = None

        # Define/remember the state/coordinates (x,y,theta) of the node
        self.x = x
        self.y = y
        
    # Node Utilities
    # To print the node
    def __repr__(self):
        return("<Node %5.2f,%5.2f>" % (self.x, self.y))
    
    # Compute/create an intermediate node for checking the local planner
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                self.y + alpha * (other.y - self.y))
    
    # Return a tuple of coordinates to compute Euclidean distances
    def coordinates(self):
        return(self.x, self.y)
    
    # Compute the relative distance Euclidean distance to another node
    def distance(self, other):
        return dist(self.coordinates(), other.coordinates())
    
    # Collision functions:
    # Check whether in free space
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        return walls.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this node connects to another node

    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return walls.disjoint(line)

# RRT Functions
def rrt(startnode, goalnode):
# def rrt(startnode, goalnode, visual):
    t_rrt_start = time.time()
    startnode.parent = None
    tree = [startnode]

    # Add a new node to the tree
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)

    steps = 0
    while True:
        # Select a random node biased towards the goal
        p = 0.3
        # The targetnode will be the goal node p% of the time and a random node (1-p)% of the time
        if random.random() <= p:
            targetnode = goalnode
        else:
            targetnode = Node(random.uniform(xmin, xmax), random.uniform(ymin, ymax))

        # Determine the distances from each node in the RRT tree to the targetnode
        distances = np.array([node.distance(targetnode) for node in tree])
        index   = np.argmin(distances) # returns the index of the node (tree position) that is closest to the targetnode
        nearnode = tree[index] # finds the closest node in the tree to the targetnode
        d = distances[index] # the distance between the nearnode and the target node

        # Get the nextnode (a node that is between the nearnode and the targetnode)
        x_next = nearnode.x + STEP_SIZE/d * (targetnode.x - nearnode.x)
        y_next = nearnode.y + STEP_SIZE/d * (targetnode.y - nearnode.y)
        
        nextnode = Node(x_next, y_next)

        # Check that nextnode is valid
        if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
            addtotree(nearnode, nextnode)

            # Check for connection to goal if nextnode is within a stepsize
            if (nextnode.distance(goalnode) <= STEP_SIZE) and (nextnode.connectsTo(goalnode)):
                addtotree(nextnode, goalnode)
                t_rrt_end = time.time()
                t_rrt = t_rrt_end - t_rrt_start
                print(f'Time taken to find goal: {t_rrt}')
                break

        # Check whether step/node criterion met
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            t_rrt_end = time.time()
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            t_rrt = t_rrt_end - t_rrt_start
            print(f'Time taken for planner to fail lol: {t_rrt}')
            return None
        
    # Create the path
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    print(path)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
    return path


# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1

# MAIN
def main():
    print('Running with step size ', STEP_SIZE, ' and up to ', NMAX, ' nodes.')

    # Create the start and goal nodes
    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    # Call the RRT function
    print('Running RRT')
    path = rrt(startnode, goalnode)
    # path = rrt(startnode, goalnode)

    # If unable to connect path, note this
    if not path:
        print('UNABLE TO FIND A PATH')
        return
    
    # Otherwise, show the path created
    print('Showing the raw path')


    # Post-process the path
    PostProcess(path)
        
    # Show the post-processed path
    print('Showing the post-processed path')
    print(path)

    with open('/home/brittany/IRIS_env_drake/maze1_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for ele in path:
            writer.writerow([ele.x, ele.y])

    # waypoints = [[node.x, node.y, 0.1] for node in path]
    # return waypoints

if __name__ == "__main__":
    main()