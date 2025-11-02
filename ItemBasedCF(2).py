import numpy as np

class ItemBasedCF():

    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        self.numUser, self.numItem = self.ratingMatrix.shape


    def calculateCosineSimilarity(self, itemIndex1, itemIndex2):

        rating_1 = self.ratingMatrix[:, itemIndex1]
        rating_2 = self.ratingMatrix[:, itemIndex2]

        dot_product = np.dot(rating_1, rating_2)
        norm_a = np.linalg.norm(rating_1)
        norm_b = np.linalg.norm(rating_2)

        similarity = dot_product / (norm_a * norm_b)

        return similarity

    def getCosineSimilarityMatrix(self):

        matrix = np.zeros((self.numItem, self.numItem))

        for i in range(self.numItem):
            for j in range(self.numItem):
                matrix[i,j] = self.calculateCosineSimilarity(i,j)

        return matrix


# zero was put for missing values
ratingMatrix = np.array([[7, 6, 7, 4, 5, 4],
                         [7, 6, 0, 4, 3, 4],
                         [0, 3, 3, 1, 1, 0],
                         [1, 2, 2, 3, 3, 4],
                         [1, 0, 1, 2, 3, 3]])

CF = ItemBasedCF(ratingMatrix)
similarityMatrix = CF.getCosineSimilarityMatrix()

print("Similarity Matrix for Item-based Collaborative Filtering:")
print(similarityMatrix)
