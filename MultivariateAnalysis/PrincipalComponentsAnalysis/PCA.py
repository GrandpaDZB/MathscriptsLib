import numpy as np



class PCA:
    def __init__(self, expect_dim):
        self.X = []
        self.D = []
        self.C = []
        self.expect_dim = expect_dim
        pass

    # X是x的行向量级联
    def import_X_fromVar(self, X):
        self.X = X
        return

    def count_D(self):
        Y = np.matmul(self.X.T, self.X)
        (eigen_values, eigen_vectors) = np.linalg.eig(Y)
        index = np.argmax(eigen_values)
        self.D = np.matrixlib.matrix(eigen_vectors[:,index]).T
        eigen_values = np.delete(eigen_values, index)
        eigen_vectors = np.delete(eigen_vectors, index, axis = 1)
        for _ in range(self.expect_dim - 1):
            index = np.argmax(eigen_values)
            self.D = np.concatenate((self.D, np.matrixlib.matrix(eigen_vectors[:,index]).T), axis = 1)
            np.delete(eigen_values, index)
            np.delete(eigen_vectors, index, axis = 1)
        print("Transform Matrix D, for c = D^T*x: ")
        print(self.D)

    def count_C(self):
        #self.C = np.arange(self.expect_dim*self.X.shape[1]).reshape(self.expect_dim, self.X.shape[1])
        self.C = np.matmul(self.D.T, self.X.T)
        print("LowerDim Vector Matrix C:")
        print(self.C)

pca = PCA(2)
X = np.random.randint(1,10,size=(3,5))
pca.import_X_fromVar(X)
pca.count_D()
pca.count_C()