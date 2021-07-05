from os import error
import numpy as np
from numpy.lib.function_base import gradient
import xlrd



class LogisticRegression():
    def __init__(self):
        self.w = []
        pass

    def data_loader(self, file, sheet_index=0):
        data = xlrd.open_workbook(file)
        table = data.sheet_by_index(sheet_index)
        rows = table.nrows
        cols = table.ncols
        X = []
        for i in range(rows):
            X.append(table.row_values(i)[:-1])
        Y = table.col_values(cols - 1)
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)

    def sigmoid(self, x):
        if x > 700:
            return 1
        if x < -700:
            return 0
        return 1/(1+np.math.exp(-x))
    
    def predict(self, x):
        return self.sigmoid(np.dot(self.w, x))

    def compute_gradient(self):
        self.gradient = np.dot(self.fx-self.Y_train, self.X_train)
        return self.gradient


    def fit(self, X=[], Y=[], iter=10000, step=0.1):
        # input data check
        if X == [] or Y == []:
            X = self.X_train
            Y = self.Y_train
        # initialization
        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.w = np.ones((1,self.dim))
        self.fx = np.dot(self.w, X.T)[0]
        for i in range(self.n):
            self.fx[i] = self.sigmoid(self.fx[i])
    
        # gradient descend
        for iteration in range(iter):
            accuracy = 0
            self.w -= step*self.compute_gradient()
            self.fx = np.dot(self.w, X.T)[0]
    
            for i in range(self.n):
                self.fx[i] = self.sigmoid(self.fx[i])
                if (self.fx[i] >= 0.5 and Y[i] > 0.9) or (self.fx[i] < 0.5 and Y[i] < 0.1):
                    accuracy += 1
            accuracy = accuracy/self.n
            print(f'iteration = {iteration+1}/{iter}, accuracy = {accuracy}') 
        

model = LogisticRegression()
model.data_loader("./test_data.xls")
model.fit(iter=10000)
        
        
