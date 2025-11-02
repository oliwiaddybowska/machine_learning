import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

def decisionTreeClassifier():
    df = pd.read_csv("IRIS.csv")

    training,test = train_test_split(df, test_size=0.3)

    train_X = training[["sepal_length","sepal_width","petal_length","petal_width"]]
    train_y = training["species"]

    test_X = test[["sepal_length","sepal_width","petal_length","petal_width"]]
    test_y = test["species"]

    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction,test_y)
    print("The accuracy of the Decision Tree is:", accuracy)

    tree.plot_tree(model,filled=True)
    plt.show()
decisionTreeClassifier()
