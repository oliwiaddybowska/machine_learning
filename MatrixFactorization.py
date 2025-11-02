import numpy as np
import matplotlib.pyplot as plt

class MF:
    def __init__(self, R, K, alpha, tol):
        self.R = R  # user-item rating matrix
        self.num_users, self.num_items = R.shape  # number fo users & items
        self.K = K  # number of latent dimension
        self.alpha = alpha  # learning rate
        self.tol = tol

        self.U = np.random.rand(self.num_users, self.K)
        self.V = np.random.rand(self.num_items, self.K)

    def calcTotalErrors(self):
        E = self.R - np.dot(self.U, self.V.T)

        error = 0
        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.R[i, j] > 0:
                    error += E[i, j] ** 2

        return error

    def train(self):
        itr = 0
        self.lstError = []

        error = self.calcTotalErrors()
        self.lstError.append(error)

        while error > self.tol and itr < 1000:
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if self.R[i, j] > 0:
                        e = self.R[i, j] - np.dot(self.U[i], self.V[j])
                        new_ui = self.U[i] + self.alpha * e * self.V[j]
                        new_vj = self.V[j] + self.alpha * e * self.U[i]

                        self.U[i] = new_ui
                        self.V[j] = new_vj

            itr += 1
            error = self.calcTotalErrors()
            self.lstError.append(error)

            print(f"Current iter: {itr}, with Error {error}")

    def plotErrorTrajectory(self):
        plt.plot(self.lstError, marker='o')
        plt.title('Error Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.show()

    def saveModel(self):
        np.save('user_matrix.npy', self.U)
        np.save('item_matrix.npy', self.V)

    def loadModel(self):
        self.U = np.load('user_matrix.npy')
        self.V = np.load('item_matrix.npy')

    def estimate_rating(self, i, j):
        estimated_ratings = np.dot(self.U[i], self.V[j].T)

        return estimated_ratings


R = np.array([[3, 0, 1, 3, 1], [1, 0, 4, 1, 0], [3, 1, 0, 3, 1], [0, 3, 0, 4, 4]])

# Hyperparameters
K = 2  # Latent factor dimension
alpha = 0.01  # Learning rate
tol = 0.0001  # Convergence threshold

mf_model = MF(R, K, alpha, tol)
mf_model.train()
mf_model.plotErrorTrajectory()
mf_model.saveModel()

estimated_ratings = mf_model.estimate_rating(1, 2)
print("Estimated rating:", estimated_ratings)
