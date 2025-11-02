import numpy as np


ratingMatrix = np.array([[7, 6, 7, 4, 5, 4],
                         [7, 6, 0, 4, 3, 4],
                         [0, 3, 3, 1, 1, 0],
                         [1, 2, 2, 3, 3, 4],
                         [1, 0, 1, 2, 3, 3]])

numUser, numItem = ratingMatrix.shape
print("# of Users:", numUser)
print("# of Items:", numItem)

similarityMatrix = np.zeros((numItem, numItem))
print("Empty similarity matrix:")
print(similarityMatrix)

for i in range(numItem):
    for j in range(numItem):
        rating_1 = ratingMatrix[:, i] # slices the matrix to extract all rows (:) of a specific column 𝑖.
        rating_2 = ratingMatrix[:, j]

        dot_product = np.dot(rating_1, rating_2) # dot product of two vectors
        norm_a = np.linalg.norm(rating_1) # Euclidean norm of the vector
        norm_b = np.linalg.norm(rating_2)

        similarityMatrix[i,j] = dot_product / (norm_a * norm_b)

print("Similarity Matrix for Item-based Collaborative Filtering:")
print(similarityMatrix)

