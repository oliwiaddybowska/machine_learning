import numpy as np
import matplotlib.pyplot as plt

R = np.array([[3, 0, 1, 3, 1], [1, 0, 4, 1, 0], [3, 1, 0, 3, 1], [0, 3, 0, 4, 4]])

num_users, num_items = R.shape
K = 2

U = np.random.rand(num_users, K)
V = np.random.rand(num_items, K)

# print("Initial user feature matrix:")
# print(U)
#
# print("Initial item feature matrix:")
# print(V)

R_hat = np.dot(U, V.T)

# print("Reconstructed rating matrix:")
# print(R_hat)


def calcTotalErrors(R, U, V):
    E = R - np.dot(U, V.T)

    error = 0
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                error += E[i, j] ** 2
    error = error / 2

    return error


error = calcTotalErrors(R, U, V)
# print("Current error", error)

tol = 0.00001
alpha = 0.01
IstError = []

while error > tol:
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                e = R[i, j] - np.dot(U[i], V[j])
                new_ui = U[i] + alpha * e * V[j]
                new_vj = V[j] + alpha * e * U[i]

                U[i] = new_ui
                V[j] = new_vj

    error = calcTotalErrors(R, U, V)
    IstError.append(error)

print("Trained User Feature Matrix:")
print(U)

print("Trained Item Feature Matrix:")
print(V)

print("Reconstructed Rating Matrix:")
print(np.dot(U, V.T))

plt.plot(IstError, marker='o')
plt.title('Error Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()

np.save('user_matrix.npy', U)
np.save('item_matrix.npy', V)

loaded_U = np.load('user_matrix.npy')
loaded_V = np.load('item_matrix.npy')


def estimate_rating(U, V, userIndex, itemIndex):
    estimated_ratings = np.dot(U[userIndex], V[itemIndex].T)

    return estimated_ratings


estimateRating = estimate_rating(U, V, 2, 3)
print("Estimated rating of user 3 for item 4:", estimateRating)
