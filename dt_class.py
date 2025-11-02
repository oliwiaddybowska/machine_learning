import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

'''def calcAverageAccuracy():
    df = pd.read_csv("IRIS.csv")

    iterationNum = 10
    lstAccuracy = []
    for i in range(iterationNum):
        train,test = train_test_split(df,test_size=0.3)
        train_X = train[["sepal_length","sepal_width","petal_length","petal_width"]]
        train_y = train[["species"]]
        test_X = test[["sepal_length","sepal_width","petal_length","petal_width"]]
        test_y = test[["species"]]

        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train_X,train_y)
        prediction = model.predict(test_X)
        accuracy = metrics.accuracy_score(prediction,test_y)
        lstAccuracy.append(accuracy)

    avgAccuracy = sum(lstAccuracy)/iterationNum

    print(f"Average accuracy over {iterationNum} iterations: {avgAccuracy}")

calcAverageAccuracy()'''

def getAverageAccuracy(testSize):
    df = pd.read_csv("IRIS.csv")

    iterationNum = 10
    lstAccuracy = []
    for i in range(iterationNum):
        train, test = train_test_split(df, test_size=testSize)
        train_X = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        train_y = train[["species"]]
        test_X = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        test_y = test[["species"]]

        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train_X, train_y)
        prediction = model.predict(test_X)
        accuracy = metrics.accuracy_score(prediction, test_y)
        lstAccuracy.append(accuracy)

    avgAccuracy = sum(lstAccuracy) / iterationNum
    return avgAccuracy


test_sizes = np.arange(0.1,0.8,0.05)
accuracy_score = []

for testSize in test_sizes:
    avgAccuracy = getAverageAccuracy(testSize)
    accuracy_score.append(avgAccuracy)

# plotting the results
plt.plot(test_sizes,accuracy_score,marker='o')
plt.xlabel('Test Size')
plt.ylabel('Average Accuracy')
plt.show()
