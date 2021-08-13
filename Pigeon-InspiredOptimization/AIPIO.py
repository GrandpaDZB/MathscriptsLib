from math import sqrt
import numpy as np


class PIO():
    def __init__ (self, dim, target_func, restrict_func = None, 
                  R=0.3, N=40, 
                  random_scale=1.0, random_bias=0.0):
        self.dim = dim
        self.target_func = target_func
        self.restrict_func = restrict_func
        # the map and compass factor
        self.R = R
        # number of pigeons
        self.N = N
        
        # initialization
        self.x = []
        self.v = []
        self.Xp = []
        for i in range(N):
            print(f'Initialize x{i+1}')
            tmp = (np.random.rand(dim)-0.5)*2*random_scale+random_bias
            while restrict_func != None and restrict_func(tmp) == False:
                tmp = (np.random.rand(dim)-0.5)*2*random_scale+random_bias
            self.x.append(tmp)
            self.Xp.append(tmp)
            self.v.append(np.zeros(dim))
        self.Xg = self.x[0]
        self.Fit = np.zeros(N)
        pass

    def restriction_check(self):
        for i in range(self.N):
            if self.restrict_func(self.x[i]) == False:
                return (False, i)
        return (True, 0)

    def compute_candidate(self):
        Average = np.mean(self.Xp, 0)
        Std = np.std(self.Xp, 0)
        candidate = np.random.normal(Average, Std)
        return candidate

    # Nc: the maximum number of generations that the map and compass operation
    # Nl: the maximum number of generations that the landmark operation
    def run(self, Nc=1000, Nl=1000):
        T = 0
        have_candidate = False
        # Map and compass operation
        for iteration in range(Nc):
            is_chg = 0
            if have_candidate:
                self.v_cand = []
                self.x_cand = []
                self.f_cand = []
            for i in range(self.N):
                if have_candidate:
                    self.v_cand.append(self.v[i]*np.math.exp(-self.R*i)+np.random.rand()*(candidate-self.x[i]))
                    if self.restrict_func != None and self.restrict_func(self.x[i] + self.v_cand[-1][0]) == False:
                        self.x_cand.append(self.x[i])
                    else:
                        self.x_cand.append(self.x[i] + self.v_cand)
                self.v[i] = self.v[i]*np.math.exp(-self.R*i)+np.random.rand()*(self.Xg-self.x[i])  
                if self.restrict_func != None and self.restrict_func(self.x[i] + self.v[i]) == False:
                    pass
                else:
                    self.x[i] += self.v[i]
                if have_candidate:
                    try:
                        self.f_cand.append(self.target_func(self.x_cand[-1][0]))
                    except:
                        self.f_cand.append(self.target_func(self.x_cand[-1]))
                self.Fit[i] = self.target_func(self.x[i])
                min_f = self.Fit[i]
                if min_f < self.target_func(self.Xp[i]):
                    self.Xp[i] = self.x[i]
                    is_chg += 1
                if min_f < self.target_func(self.Xg):
                    self.Xg = self.x[i]
            if have_candidate:
                Competitiveness_gbest = np.sum(self.Fit)
                Competitiveness_candidate = np.sum(self.f_cand)
                # if Competitiveness_candidate < Competitiveness_gbest:
                self.Xg = candidate
                    # print("==============================")
                    # print("change to a candidate")
                    # print("==============================")
                have_candidate = False

            if is_chg == 0 and T < 10:
                T += 1
            Prob_adjust = (np.math.exp(T)-1)/(np.math.exp(10)-1)
            if Prob_adjust > np.random.rand():
                # change to a candidate
                candidate = self.compute_candidate()
                have_candidate = True
            print(f'Nc Iteration = {iteration}')
            print(f'min_f = {self.target_func(self.Xg)}')
            print(f'var = {np.var(np.array(self.x), 0)}')
        
        # Landmark operations
        for iteration in range(Nl):
            new_x = []
            for i in range(self.N):
                min_f = self.target_func(self.x[i])
                min_num = i
                for j in range(i+1, self.N):
                    tmp_min_f = self.target_func(self.x[j]) 
                    if tmp_min_f < min_f:
                        min_f = tmp_min_f 
                        min_num = j
                tmp_f = self.x[i]
                self.x[i] = self.x[min_num]
                self.x[min_num] = tmp_f
                new_x.append(self.x[i])
                if i == int(self.N/2):
                    break
            self.N = int(self.N/2) + 1
            self.x = new_x
            Xc = np.mean(new_x, 0)

            for i in range(self.N):
                tmp_x = self.x[i] + np.random.rand()*(Xc-self.x[i])
                if self.restrict_func != None and self.restrict_func(tmp_x) == False:
                    pass
                else:
                    self.x[i] = tmp_x
            print(f'Nl Iteration = {iteration}')
            print(f'min_f = {self.target_func(self.x[0])}')
            print(f'var = {np.var(np.array(self.x), 0)}')
            if self.N == 2:
                break

        
        
def shubert(x):
    z1 = (1 * np.cos((1 + 1) * x[0] + 1)) + (2 * np.cos((2 + 1) * x[0] + 2)) + (3 * np.cos((3 + 1) * x[0] + 3)) + (
            4 * np.cos((4 + 1) * x[0] + 4)) + (5 * np.cos((5 + 1) * x[0] + 5))
    z2 = (1 * np.cos((1 + 1) *x[1] + 1)) + (2 * np.cos((2 + 1) *x[1] + 2)) + (3 * np.cos((3 + 1) *x[1] + 3)) + (
            4 * np.cos((4 + 1) *x[1] + 4)) + (5 * np.cos((5 + 1) *x[1] + 5))
    return -(z1*z2)
def target_func(x):
    return 0.5+((np.math.pow(np.math.sin(sqrt(x[0]*x[0]+x[1]*x[1])),2)-0.5)/np.math.pow((1+0.001*(x[0]*x[0]+x[1]*x[1])),2))
def restrict_func(x):
    if abs(x[0]) > 1 or abs(x[1]) > 1:
        return False
    else:
        return True
model = PIO(2, shubert, random_scale=1, restrict_func=restrict_func)
model.run(1000,1000)




