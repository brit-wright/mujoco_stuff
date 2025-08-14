import numpy as np
from math import sin, cos, tan, sqrt

L1 = 0.213
L2 = 0.213

class NewtonRaphson:

    def __init__(self, p_curr, p_desired, q_curr, chain_base_foot, chain_base_hip):

        self.p_curr = p_curr
        self.p_desired = p_desired
        self.q_curr = q_curr

        self.chain_base_foot = chain_base_foot
        self.chain_base_hip = chain_base_hip

        self.iter_limit = 100
        
    def calc_distance(self, point1, point2):

        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    
    def create_parameters(self):

        distance = self.calc_distance(self.p_curr, self.p_desired)

        if distance <= 0.08:
            self.num_nr = 1

            self.p_desired_list = [self.p_desired]

        elif distance <= 0.16:
            self.num_nr = 2

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.5
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.5
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.5

            p_des1 = [x_des1, y_des1, z_des1]
            self.p_desired_list = [p_des1, self.p_desired]

        elif distance <= 0.24:
            self.num_nr = 3

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 1/3
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 1/3
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 1/3
            
            x_des2 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 2/3
            y_des2 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 2/3
            z_des2 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 2/3

            p_des1 = [x_des1, y_des1, z_des1]
            p_des2 = [x_des2, y_des2, z_des2]
            self.p_desired_list = [p_des1, p_des2, self.p_desired]

        elif distance <= 0.32:
            self.num_nr = 4

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.25
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.25
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.25

            x_des2 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.5
            y_des2 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.5
            z_des2 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.5

            x_des3 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.75
            y_des3 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.75
            z_des3 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.75

            p_des1 = [x_des1, y_des1, z_des1]
            p_des2 = [x_des2, y_des2, z_des2]
            p_des3 = [x_des3, y_des3, z_des3]
            self.p_desired_list = [p_des1, p_des2, p_des3, self.p_desired]

   # # lazy fix
        # if self.cycle == 0:
        #     pd_leg = self.pdlast
        #     self.stance1 = pd_leg
        # else:
        #     pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
        #     self.stance1 = pd_leg

        # # smarter fix
        # pd_leg = double_stance1(t, self.double_stance_time, 0, self.deltx * (self.double_stance_time/self.cycle_len))
        # self.stance1 = pd_leg     elif distance <= 0.40:
            
            self.num_nr = 5

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.2
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.2
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.2

            x_des2 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.4
            y_des2 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.4
            z_des2 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.4

            x_des3 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.6
            y_des3 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.6
            z_des3 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.6

            x_des4 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.8
            y_des4 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.8
            z_des4 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.8

            p_des1 = [x_des1, y_des1, z_des1]
            p_des2 = [x_des2, y_des2, z_des2]
            p_des3 = [x_des3, y_des3, z_des3]
            p_des4 = [x_des4, y_des4, z_des4]
            self.p_desired_list = [p_des1, p_des2, p_des3, p_des4, self.p_desired]

        elif distance <= 0.48:
            self.num_nr = 6

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 1/6
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 1/6
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 1/6

            x_des2 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 2/6
            y_des2 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 2/6
            z_des2 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 2/6

            x_des3 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 0.5
            y_des3 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 0.5
            z_des3 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 0.5

            x_des4 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 4/6
            y_des4 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 4/6
            z_des4 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 4/6

            x_des5 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 5/6
            y_des5 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 5/6
            z_des5 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 5/6

            p_des1 = [x_des1, y_des1, z_des1]
            p_des2 = [x_des2, y_des2, z_des2]
            p_des3 = [x_des3, y_des3, z_des3]
            p_des4 = [x_des4, y_des4, z_des4]
            p_des5 = [x_des5, y_des5, z_des5]
            self.p_desired_list = [p_des1, p_des2, p_des3, p_des4, p_des5, self.p_desired]

        else:
            self.num_nr = 9

            x_des1 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 1/9
            y_des1 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 1/9
            z_des1 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 1/9

            x_des2 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 2/9
            y_des2 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 2/9
            z_des2 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 2/9

            x_des3 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 3/9
            y_des3 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 3/9
            z_des3 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 3/9

            x_des4 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 4/9
            y_des4 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 4/9
            z_des4 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 4/9

            x_des5 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 5/9
            y_des5 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 5/9
            z_des5 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 5/9

            x_des6 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 6/9
            y_des6 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 6/9
            z_des6 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 6/9

            x_des7 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 7/9
            y_des7 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 7/9
            z_des7 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 7/9

            x_des8 = self.p_curr[0] + (self.p_desired[0] - self.p_curr[0]) * 8/9
            y_des8 = self.p_curr[1] + (self.p_desired[1] - self.p_curr[1]) * 8/9
            z_des8 = self.p_curr[2] + (self.p_desired[2] - self.p_curr[2]) * 8/9

            p_des1 = [x_des1, y_des1, z_des1]
            p_des2 = [x_des2, y_des2, z_des2]
            p_des3 = [x_des3, y_des3, z_des3]
            p_des4 = [x_des4, y_des4, z_des4]
            p_des5 = [x_des5, y_des5, z_des5]
            p_des6 = [x_des6, y_des6, z_des6]
            p_des7 = [x_des7, y_des7, z_des7]
            p_des8 = [x_des8, y_des8, z_des8]
            self.p_desired_list = [p_des1, p_des2, p_des3, p_des4, p_des5, p_des6, p_des7, p_des8, self.p_desired]

        return self.num_nr, self.p_desired_list
    
    def do_newton_raphson(self, q_curr, p_des):

        counter = 0

        while counter < self.iter_limit:

            counter += 1
            
            t0 = q_curr[0]
            t1 = q_curr[1]
            t2 = q_curr[2]

            # Jacob = np.array([[0,                                                                                                                     -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
            #           [0.0955*tan(t0)*(1/cos(t0)) + L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0) - 0.0955*tan(t0)*(1/cos(t0)), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
            #           [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) - 0.0955*tan(t0)*sin(t0) + 0.0955*(1/cos(t0)),                              L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])

            Jacob = np.array([[0,                                                       -L1*cos(t1)-L2*cos(t1+t2),                 -L2*cos(t1+t2)],
                      [L1*cos(t1)*cos(t0) + L2*cos(t1+t2)*cos(t0) - 0.0955*sin(t0), -L1*sin(t0)*sin(t1)-L2*sin(t0)*sin(t1+t2), -L2*sin(t0)*sin(t1+t2)],
                      [L1*cos(t1)*sin(t0) + L2*cos(t1+t2)*sin(t0) + 0.0955*cos(t0), L1*sin(t1)*cos(t0) + L2*cos(t0)*sin(t1+t2), L2*cos(t0)*sin(t1+t2)]])
            
            # calculate the current foot position based on the current joint positions
            p_base_to_foot_curr = self.chain_base_foot.fkin([t0, t1, t2])[0]
            p_base_to_hip_curr = self.chain_base_hip.fkin([t0])[0]

            p_hip_to_foot_curr = p_base_to_foot_curr - p_base_to_hip_curr

            dist = self.calc_distance(p_hip_to_foot_curr, p_des)
            # print(f'dist is {dist}')

            error = p_des - p_hip_to_foot_curr

            if np.linalg.norm(error) <= 1e-3:
                # print(f'Solution successfully found in {counter} iterations')
                return q_curr
            
            alpha = 1.0

            qnext = q_curr + alpha * np.linalg.pinv(Jacob) @ error

            q_curr = qnext

            # print(f'Current joint angles are {q_curr}')

        if counter >= self.iter_limit:
            # print(f'Iteration limit reached')
            return [None]


    def call_newton_raphson(self):

        num_calls, p_des_list = self.create_parameters()

        for n in range(num_calls):

            p_des = p_des_list[n]

            self.q_curr = self.do_newton_raphson(self.q_curr, p_des)

            if None in self.q_curr:
                # print(f'We flopped during iteration {n} out of {num_calls}, chief :/')
                break
        
        # print(f'Joint angles found from Newton-Raphson method: {self.q_curr}')
        
        return self.q_curr

        


