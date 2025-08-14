# imports
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

# create a class for Bezier curve stuff
class Bezier3D():

    def __init__(self, xstart, xend, ground, step_height, ystart, yend):
        
        self.xstart = xstart
        self.xend = xend
        self.ground = ground
        self.step_height = step_height
        self.ystart = ystart
        self.yend = yend

        x_inc = (xstart + xend)/15
        x_mid = (xstart + xend)/2

        y_inc = (ystart + yend)/15
        y_mid = (ystart + yend)/2


        # create the control points
        self.ctrl_pts = [np.array([xstart, ystart, ground]), np.array([x_inc, y_inc, ground]), np.array([x_mid, y_mid, step_height]),
                         np.array([xend-x_inc, yend-y_inc, ground]), np.array([xend, yend, ground])]
        
        self.ctrl_num = len(self.ctrl_pts)

    def get_coeffs(self, n, idx):
        coeff = factorial(n)/(factorial(idx) * factorial(n-idx))
        return coeff
    
    def get_zmax_curve(self):

        ans = 0
        n = self.ctrl_num - 1
        
        for idx in range(n+1):
            term = self.get_coeffs(n, idx) * (1 - 0.5)**(n-idx) * (0.5 ** idx) * self.ctrl_pts[idx][-1]
            ans += term

        return ans
    
    def correct_heights(self, sum):
        
        self.mid = self.get_zmax_curve()

        # sum = self.ground - ((sum - self.ground)/(self.mid - self.ground)) * (self.ground - self.step_height)

        sum[2] = (sum[2] - self.ground)/(self.mid - self.ground) * self.step_height

        return sum
        
    def create_bezier(self, curr):
        
        sum = 0
        n = self.ctrl_num - 1
        
        for idx in range(n+1):
            term = self.get_coeffs(n, idx) * (1 - curr)**(n-idx) * (curr ** idx) * self.ctrl_pts[idx]
            sum += term

        if (self.ground != self.step_height):
            sum = self.correct_heights(sum)
        
        return sum
    
def main():
    
    # create the Bezier function object
    bez = Bezier3D(0.0, 0.8, -0.3, 0.1, 0.0, 0.015)

    # find the terms of the bezier function
    x_list = np.arange(0, 0.8+0.01, 0.01)
    vals = []

    for x_val in x_list:

        x_curr = (x_val - 0.0)/(0.8 - 0.0) # normalize the x_curr val to be between 0 and 1

        val = bez.create_bezier(x_curr)
        print(f'val is {val}')
        vals.append(val)

    # plotting

    ax = plt.axes(projection='3d')

    x_vals = [val[0] for val in bez.ctrl_pts]
    y_vals = [val[1] for val in bez.ctrl_pts]
    z_vals = [val[2] for val in bez.ctrl_pts]

    x_vals_bez = [val[0] for val in vals]
    y_vals_bez = [val[1] for val in vals]
    z_vals_bez = [val[2] for val in vals]

    # plot the control points
    # ax.scatter3D(x_vals, y_vals, z_vals)

    # plot the bezier function points
    ax.plot3D(x_vals_bez, y_vals_bez, z_vals_bez)

    # plot the connections between the control points
    # for i in range(len(bez.ctrl_pts)-1):
    #     x_vals = [bez.ctrl_pts[i][0], bez.ctrl_pts[i+1][0]]
    #     y_vals = [bez.ctrl_pts[i][1], bez.ctrl_pts[i+1][1]]
    #     z_vals = [bez.ctrl_pts[i][2], bez.ctrl_pts[i+1][2]]
    #     ax.plot3D(x_vals, y_vals, z_vals, 'g-')

    plt.show()

if __name__=="__main__":
    main()
