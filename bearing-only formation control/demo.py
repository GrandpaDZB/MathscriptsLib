import numpy as np
import math as m
import matplotlib.pyplot as plt


def eye_matrix(list_m):
    n = 0
    index = 0
    for each in list_m:
        n += each.shape[0]
    eye_m = np.zeros((n,n))
    for each in list_m:
        m = each.shape[0]
        for i in range(m):
            for j in range(m):
                eye_m[index+i, index+j] = each[i,j]
        index += m
    return eye_m
        

class controller:
    def __init__(self):
        self.dim = 2
        self.Id = np.eye(2)
        self.n = 4
        self.m = 5
        self.H = []
        self.H_bar = []
        self.p = []
        self.p_init = []
        self.e = []
        self.e_norm = []
        self.g = []
        self.g_exp = []
        pass

    # matrix H would form G
    def import_H(self, H):
        self.H = H
        self.H_bar = np.kron(H,self.Id)
        return

    # import intial p 
    def import_initial_p(self, p):
        self.p = p
        self.p_init = p
        return

    # compute e after import H and p
    def compute_e(self, p):
        e = np.matmul(self.H_bar, np.transpose(p))
        g = np.copy(e)
        e_norm = []
        for i in range(self.m):
            norm_e = np.linalg.norm([e[2*i],e[2*i+1]])
            e_norm.append(norm_e)
            g[2*i] = e[2*i]/norm_e
            g[2*i+1] = e[2*i+1]/norm_e
        return [e, e_norm, g]

    def P_x(self, x):
        d = x.size
        x = np.reshape(x, [d,1])
        P_x = np.eye(d) - (np.matmul(x,np.transpose(x)))/np.linalg.norm(x)/np.linalg.norm(x)
        return P_x

    # compute bearing rigidity matrix
    def R_p_and_diagP(self, p):
        tmp_diag = []
        [e, e_norm, g] = self.compute_e(p)
        for i in range(self.m):
            gk = np.array([g[2*i], g[2*i+1]])
            Pgk = self.P_x(gk)
            tmp_diag.append(Pgk/e_norm[i])
        tmp_diag = eye_matrix(tmp_diag)
        R_p = np.matmul(tmp_diag, self.H_bar)
        return [R_p, tmp_diag]

    def import_g_exp(self, g_exp):
        self.g_exp = np.copy(g_exp)
        for i in range(self.m):
            norm_g = np.linalg.norm([g_exp[2*i],g_exp[2*i+1]])
            self.g_exp[2*i] = g_exp[2*i]/norm_g
            self.g_exp[2*i+1] = g_exp[2*i+1]/norm_g
        return

    def compute_v(self, diag_P):
        v = self.H_bar.T@diag_P@self.g_exp
        return v


model = controller()

# import H
H = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1],[0,-1,0,1],[0,0,-1,1]])
model.import_H(H)

# build random initial p in a circle
rand_r = np.random.random(4)*5.0
rand_theta = np.random.random(4)*3.1415926*2
p = []
for i in range(4):
    p.append(rand_r[i]*m.cos(rand_theta[i]))
    p.append(rand_r[i]*m.sin(rand_theta[i]))
p = np.array(p)    
# p = np.array([0,1,1,1,0,0,1,0])
model.import_initial_p(p)

# compute e
[model.e, model.e_norm, model.g] = model.compute_e(p)
[R_p,diag_P] = model.R_p_and_diagP(p)

# import p_exp
g_exp = np.array([1,0,0,-1,1,-1,0,-1,1,0], dtype='float')
model.import_g_exp(g_exp)

diff = np.sum(np.abs(model.g-model.g_exp))

# simulation
epsilon = 1.0
history = [np.copy(model.p)]
while diff > 0.1:
    v = -model.compute_v(diag_P)
    new_p = model.p + epsilon*v
    model.p = np.copy(new_p)
    [model.e, model.e_norm, model.g] = model.compute_e(p)
    [R_p,diag_P] = model.R_p_and_diagP(p)
    diff = np.sum(np.abs(model.g-model.g_exp))
    history.append(np.copy(model.p))
    print(f'diff={diff}')
    


