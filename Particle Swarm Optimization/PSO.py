import numpy as np
import math

class PSO:
    def __init__(self, dim, target_func, restrict_func = None, N=40, w=0.5, c1=0.146, c2=0.146, random_scale=1, random_bias=0):
        self.dim = dim
        self.target_func = target_func
        self.restrict_func = restrict_func
        self.N = N
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.x = []
        self.P_i = []
        for i in range(N):
            self.x.append((np.random.rand(dim)-0.5)*2*random_scale+random_bias)
            self.P_i.append((np.random.rand(dim)-0.5)*2*random_scale+random_bias)
            print(f'Initialize {i}')
            if restrict_func != None:
                while(restrict_func(self.x[i]) == False):
                    self.x[i] = (np.random.rand(dim)-0.5)*2*random_scale+random_bias
                    self.P_i[i] = (np.random.rand(dim)-0.5)*2*random_scale+random_bias
        self.P_g = self.P_i[0]
        pass

    def run(self, iter = 1000, sigma='scale',scale=1.0, interval=(0.0,1.0), func=None):
        v = []
        for i in range(self.N):
            v.append(np.zeros(self.dim))
        for iteration in range(iter):
            for i in range(self.N):
                v_i = self.compute_velocity(v[i], self.x[i], self.P_i[i])
                x_tmp = self.compute_x(self.x[i], v_i, sigma, scale, interval, func)
                if self.restrict_func != None and self.restrict_func(x_tmp) == False:
                    pass
                else:
                    self.x[i] = x_tmp
                tmp_f = self.target_func(self.x[i])
                if tmp_f < self.target_func(self.P_i[i]):
                    self.P_i[i] = self.x[i]
                if tmp_f < self.target_func(self.P_g):
                    self.P_g = self.x[i]
            print(f'iteration = {iteration+1}')
            print(f'min_fx = {self.target_func(self.P_g)}')
            print(f'var_x = {np.var(np.array(self.x), 0)}')        
        pass

    def compute_velocity(self, v_i, x_i, P_i):
        v_i = self.w*v_i + self.c1*(P_i-x_i)*np.random.rand() + self.c2*(self.P_g-x_i)*np.random.rand()
        return v_i

    def compute_x(self, x_i, v_i_next, sigma, scale, interval, func):
        if sigma == 'func':
            v = func(v_i_next)
        elif sigma == 'scale':
            v = self.sigma_scale(v_i_next, scale)
        elif sigma == 'interval':
            v = self.sigma_interval(v_i_next, interval)
        else:
            v = v_i_next
        return x_i + v

    def sigma_scale(self, v, scale):
        return scale*v

    def sigma_interval(self, v, interval):
        for i in range(self.dim):
            if v[i] < interval[0]:
                v[i] = interval[0]
            elif v[i] > interval[1]:
                v[i] = interval[1]
            else:
                pass
        return v



def shubert(x):
    z1 = (1 * np.cos((1 + 1) * x[0] + 1)) + (2 * np.cos((2 + 1) * x[0] + 2)) + (3 * np.cos((3 + 1) * x[0] + 3)) + (
            4 * np.cos((4 + 1) * x[0] + 4)) + (5 * np.cos((5 + 1) * x[0] + 5))
    z2 = (1 * np.cos((1 + 1) *x[1] + 1)) + (2 * np.cos((2 + 1) *x[1] + 2)) + (3 * np.cos((3 + 1) *x[1] + 3)) + (
            4 * np.cos((4 + 1) *x[1] + 4)) + (5 * np.cos((5 + 1) *x[1] + 5))
    return -(z1*z2)

def target_func(x):
    return 0.5+((np.math.pow(np.math.sin(math.sqrt(x[0]*x[0]+x[1]*x[1])),2)-0.5)/np.math.pow((1+0.001*(x[0]*x[0]+x[1]*x[1])),2))

model = PSO(2, shubert, random_scale=1, w=0.5, N = 100)
model.run(3000, scale=0.05)