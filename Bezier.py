# imports
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

# create a class for Bezier curve stuff
class Bezier():

    def __init__(self, start, end, ground, step_height):
        
        self.start = start
        self.end = end
        self.ground = ground
        self.step_height = step_height

        inc = (start + end)/15
        mid = (start + end)/2

        # create the control points
        self.ctrl_pts = [np.array([start, ground]), np.array([inc, ground]), np.array([mid, step_height]),
                         np.array([end-inc, ground]), np.array([end, ground])]
        

        # inc1 = (start + end)/5
        # inc2 = (start + end)/4
        # inc3 = (start + end)/20
        # mid = (start + end)/2

        # # create the control points
        # self.ctrl_pts = [np.array([start, ground]), np.array([inc1, ground]), np.array([inc2, ground]), np.array([mid, step_height]),
        #                  np.array([end-inc2, ground]), np.array([end - inc1, ground]), np.array([end, ground])]

        # # create the control points
        # self.ctrl_pts = [np.array([start, ground]), np.array([inc1, ground]), np.array([inc2, ground]), np.array([inc3, ground]), np.array([mid, step_height]),
        #                  np.array([end-inc3, ground]), np.array([end-inc2, ground]), np.array([end - inc1, ground]), np.array([end, ground])]
        
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

        sum = (sum - self.ground)/(self.mid - self.ground) * self.step_height

        return sum
        
    def create_bezier(self, curr):
        
        sum = 0
        n = self.ctrl_num - 1
        
        for idx in range(n+1):
            term = self.get_coeffs(n, idx) * (1 - curr)**(n-idx) * (curr ** idx) * self.ctrl_pts[idx][-1]
            sum += term

        sum = self.correct_heights(sum)
        
        return sum
    
def main():
    # check if this works
    bez = Bezier(0.0, 0.8, -0.3, 0.1)
        
    x_list = np.arange(0, 0.8+0.01, 0.01)
    vals = []
    vals2 = []

    for x_val in x_list:

        x_curr = (x_val - 0.0)/(0.8 - 0.0)

        val = bez.create_bezier(x_curr)
        vals.append(val)

    plt.plot(x_list, vals)

    control_points = bez.ctrl_pts
    ctrl_x = [val[0] for val in control_points]
    ctrl_z = [val[1] for val in control_points]

    # print(vals[0])
    # print(vals[-1])

    bez.get_zmax_curve()

    for pt in control_points:
        plt.plot(pt[0], pt[1], '.')

    plt.plot(ctrl_x, ctrl_z)

    plt.show()

if __name__=="__main__":
    main()

# implementation notes:
# The initial z_swing equation looked something like this:
# z_swing = -x_swing * (x_swing - end)
# so at the start and end, z_swing MUST be 0, and all other values must be positive

# maybe try shifting values upwards after found?