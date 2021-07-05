import numpy as np
import random
import xlrd
import matplotlib.pyplot as plt



class SVM():
    def __init__(self, max_iter=10000, kernal_type='linear', C=1.0, epsilon=0.001, Gaussian_stderr = 1.0):
        self.kernels = {
            'linear': self.kernel_linear,
            'Gaussian': self.kernel_Gaussian,
        }
        self.max_iter = max_iter
        self.C = C
        self.kernal_type = kernal_type
        self.epsilon = epsilon
        self.gaussian_stderr = Gaussian_stderr;
        self.support_vector_index = []

    def data_loader(self, file_xls, sheet_index = 0, area=(0,0,0)):
        data = xlrd.open_workbook(file_xls)
        table = data.sheet_by_index(sheet_index)
        if area[0] == 0:
            rows = table.nrows
            X_array = []
            for i in range(rows-1):
                X_array.append(table.row_values(i))
            data_set = (np.array(X_array), np.array(table.row_values(rows-1)))
        else:
            X_array = []
            for i in range(area[1], area[2]):
                X_array.append(table.row_values(i))
            data_set = (np.array(X_array), np.array(table.row_values(area[2])))
        return data_set


    def randint_not_j(self, a,b,j):
        rand_r = random.randint(a,b)
        while(rand_r == j):
            rand_r = random.randint(a,b)
        return rand_r

    def compute_b(self, X, y, w):
        if self.kernal_type == 'linear':
            b_tmp = y - np.dot(w.T, X)
        else:
            b_tmp = []
            for i in range(self.n):
                b_tmp.append(self.y[i]-np.dot(self.K[i,:], (self.alpha*self.y).T))
        return np.mean(b_tmp)

    def compute_w(self, alpha, y, X):
        return np.dot(X, (alpha*y).T)

    # prediction
    def predict(self, x):
        if self.kernal_type == 'linear':
            return np.sign(np.dot(self.w.T,x)+self.b).astype(int)
        else:
            result = self.b
            for i in range(self.n):
                if self.alpha[0,i] != 0:
                    result += self.alpha[0,i]*self.y[i]*self.kernels[self.kernal_type](self.X[:,i], x)
            return np.sign(result)

    # prediction_inside
    def pred_in(self, x_index):
        if self.kernal_type == 'linear':
            return np.sign(np.dot(self.w.T,self.X[:,x_index])+self.b).astype(int)
        else:
            result = self.b
            for i in range(self.n):
                if self.alpha[0,i] != 0:
                    result += self.alpha[0,i]*self.y[i]*self.kernels[self.kernal_type](self.X[:,i], self.X[:,x_index])
            return np.sign(result)

    # prediction error
    def Err(self, x_index, y_index):
        return self.pred_in(x_index) - self.y[y_index]

    def compute_L_H(self, C, alpha_old_i, alpha_old_j, y_i, y_j):
        if(y_i != y_j):
            return [max(0, alpha_old_j - alpha_old_i),
                    min(C, C-alpha_old_i+alpha_old_j)]
        else:
            return [max(0, alpha_old_j + alpha_old_i-C),
                    min(C, alpha_old_i+alpha_old_j)]

    # Kernals
    def kernel_linear(self, x1, x2):
        return np.dot(x1.T, x2)
    def kernel_Gaussian(self, x1, x2):
        return np.math.exp(-np.linalg.norm(x1-x2)*np.linalg.norm(x1-x2)/2/self.gaussian_stderr/self.gaussian_stderr)

    # build Kernel
    def compute_Kernel(self):
        self.K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(0, self.n):
                if j <= i:
                    self.K[i,j] = self.kernels[self.kernal_type](self.X[:,i], self.X[:,j])
                else:
                    self.K[i,j] = self.K[j,i]
        return

    # model fit
    def fit(self, X, y):
        
        self.y = y
        self.X = X
        # initialization
        (dim, n) = X.shape
        self.n = n
        alpha = np.zeros((1, n))
        self.alpha = alpha
        self.compute_Kernel()
        count = 0
        while True:
            count += 1
            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return

            print(f'training {100*count/self.max_iter}% ')
            alpha_old = np.copy(alpha)
            for j in range(n):
                i = self.randint_not_j(0, n-1, j)
                y_i, y_j = y[i], y[j]
                if j > i:
                    eta = self.K[i,i]+self.K[j,j]-2*self.K[i,j]
                else:
                    eta = self.K[i,i]+self.K[j,j]-2*self.K[j,i]
                if eta == 0:
                    continue
                alpha_old_i, alpha_old_j = alpha[0,i], alpha[0,j]
                (L,H) = self.compute_L_H(self.C, alpha_old_i, alpha_old_j, y_i, y_j)

                # compute model parameters
                self.w = self.compute_w(alpha, y, X)
                self.b = self.compute_b(X, y, self.w)

                # compute Err
                Err_i = self.Err(i, i)
                Err_j = self.Err(j, j)

                # set new alpha values
                alpha[0,j] = alpha_old_j + float(y_j*(Err_i-Err_j))/eta
                alpha[0,j] = max(alpha[0,j], L)
                alpha[0,j] = min(alpha[0,j], H)

                alpha[0,i] = alpha_old_i + y_i*y_j*(alpha_old_j - alpha[0,j])
                self.alpha = alpha
                # check convergence
                diff = np.linalg.norm(alpha - alpha_old)
                if diff < self.epsilon:
                    break        
 
            # compute final model parameters
            self.w = self.compute_w(alpha, y, X)
            self.b = self.compute_b(X, y, self.w)

    def dim2_plot(self, X, y):
        plt.figure()
        for i in range(self.n):
            if y[i] == 1:
                plt.scatter(X[0,i],X[1,i], c='red', s=20)
            elif y[i] == -1:
                plt.scatter(X[0,i],X[1,i], c='green', s=20)
            else:
                pass
        min_x = int(min(X[1,:]))
        max_x = int(max(X[1,:]))
        plot_x = np.linspace(min_x, max_x, 50)
        plot_y = (-self.w[0]*plot_x-self.b)/self.w[1]
        plt.plot(plot_x, plot_y)
        plt.show()



if __name__ == "__main__":

    model = SVM(max_iter=1000, C = 1.0, kernal_type='Gaussian', Gaussian_stderr=2.0)
    (X,y) = model.data_loader('./data.xls', area=(1,0,2))
    model.fit(X,y)
    #model.dim2_plot(X,y)


    # accuracy in training set
    correct_num = 0
    y_pred = []
    for i in range(X.shape[1]):
        y_pred.append(model.predict(X[:,i]))
        if y_pred[-1] == y[i]:
            correct_num += 1
    accuracy = correct_num/X.shape[1]
    print(f'accuracy = {accuracy}')

    # accuracy in testing set
    (X_test, y_test) = model.data_loader('./data.xls', area=(1,3,5))
    correct_num = 0
    y_pred_test = []
    for i in range(X_test.shape[1]):
        y_pred_test.append(model.predict(X_test[:,i]))
        if y_pred_test[-1] == y_test[i]:
            correct_num += 1
    accuracy = correct_num/X.shape[1]
    print(f'accuracy = {accuracy}')

