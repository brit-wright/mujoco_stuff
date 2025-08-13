import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep

# PARAMETERS
STEP_SIZE = 0.5 # chose a static step size
SMAX = 20000  # maximum number of step attempts
NMAX = 10000 # maximum number of nodes

# Define the list of obstacles/objects as well as star/goal

# Define the vertices of the shapes that will be drawn to define the objects/obstacles
(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)

(xA, xB, xC, xD, xE) = ( 5, 12, 15, 18, 21)
(yA, yB, yC, yD)     = ( 5, 10, 12, 15)

xlabels = (xmin, xA, xB, xC, xD, xE, xmax)
ylabels = (ymin, yA, yB, yC, yD,     ymax)


# Draw the outer boundary/walls
outside = LineString([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                      [xmin, ymax], [xmin, ymin]])

# Draw the interior walls that the mattress will have to move around
wall1   = LineString([[xmin, yB], [xC, yB]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC], [xC, ymax]])
wall4   = LineString([[xC, yB],[xC, yA]])
wall5   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])


rect1 = LineString([[2, 18],
                    [12, 18],
                    [12, 17],
                    [2, 17],
                    [2, 18]])

rect2 = LineString([[3, 17],
                    [6, 17],
                    [6, 14],
                    [3, 14],
                    [3, 17]])

rect3 = LineString([[4, 14],
                    [5, 14],
                    [5, 6],
                    [4, 6],
                    [4, 14]])

rect4 = LineString([[3, 6],
                    [6, 6],
                    [6, 3],
                    [3, 3],
                    [3, 6]])

rect5 = LineString([[3, 3],
                    [21, 3],
                    [21, 1],
                    [3, 1],
                    [3, 3]])

rect6 = LineString([[11, 11],
                    [11, 6],
                    [12, 6],
                    [12, 11],
                    [11, 11]])

rect7 = LineString([[10, 14],
                    [10, 11],
                    [13, 11],
                    [13, 14],
                    [10, 14]])

rect8 = LineString([[11, 15],
                    [18, 15],
                    [18, 14],
                    [11, 14],
                    [11, 15]])

rect9 = LineString([[18, 16],
                    [18, 13],
                    [21, 13],
                    [21, 16],
                    [18, 16]])

rect10 = LineString([[21, 15],
                     [21, 14], 
                     [27, 14],
                     [27, 15],
                     [21, 15]])

rect11 = LineString([[27, 16],
                     [27, 13],
                     [30, 13],
                     [30, 16],
                     [27, 16]])

rect12 = LineString([[17, 10],
                     [17, 3],
                     [19, 3],
                     [19, 10],
                     [17, 10]])

rect13 = LineString([[23, 11],
                     [25, 11],
                     [25, 6],
                     [23, 6],
                     [23, 11]])

rect14 = LineString([[22, 3],
                     [22, 1],
                     [28, 1],
                     [28, 3],
                     [22, 3]])

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
walls = prep(MultiLineString([outside, rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, rect9, rect10, rect11, rect12, rect13, rect14]))
# walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart) = (21, 6)
(xgoal, ygoal) = (2, 16)

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

        # # Visualize the new node
        # visual.drawEdge(oldnode, newnode, color = 'g', linewidth = 1)
        # visual.show()

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

    # Create the figure
    # visual = Visualization()

    # Create the start and goal nodes
    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    # Visualize the start and goal nodes
    # visual.drawNode(startnode, color='orange', marker='o')
    # visual.drawNode(goalnode, color='purple', marker='o')
    # visual.show('Showing basic world') 

    # Call the RRT function
    print('Running RRT')
    # path = rrt(startnode, goalnode, visual)
    path = rrt(startnode, goalnode)

    # If unable to connect path, note this
    if not path:
        print('UNABLE TO FIND A PATH')
        return
    
    # # Otherwise, show the path created
    # visual.drawPath(path, color='r', linewidth=1)
    # visual.show('Showing the raw path')
    print('Showing the raw path')


    # Post-process the path
    PostProcess(path)
        
    # # Show the post-processed path
    # visual.drawPath(path, color='b', linewidth=2)
    # visual.show('Showing the post-processed path')
    print('Showing the post-processed path')
    print(path)

    waypoints = [[node.x, node.y, 0.1] for node in path]
    return waypoints

if __name__ == "__main__":
    main()