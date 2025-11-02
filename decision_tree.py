import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import tree

df = pd.read_csv("IRIS.csv")

print(df.head(5))
df.info()

# print summary statistics
print(df.describe())
print(df.value_counts("species"))

# sepal scatterplot
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
plt.show()

# petal scatterplot
sns.scatterplot(x="petal_length", y="petal_width", hue="species", data=df)
plt.show()

# multivariate analysis - pair plot
sns.pairplot(data=df, hue="species")
plt.show()

# feature-wise analysis - box plot
sns.boxplot(data=df, x="species", y="petal_length", hue="species")
plt.show()

df['species'] = df['species'].replace('Iris-setosa',0)
df['species'] = df['species'].replace('Iris-versicolor',1)
df['species'] = df['species'].replace('Iris-virginica',2)

# print(df.head(5))

# cross-validation method - 1
testProp = 0.3
testSize = int(testProp * len(df))

dfShuffled = df.sample(frac=1)
training = dfShuffled.iloc[testSize:]
test = dfShuffled.iloc[:testSize]

print(training.shape)
print(test.shape)

# cross-validation method - 2
training,test = train_test_split(df, test_size=0.3)
print(training.shape)
print(test.shape)

# cross-validation method - 3
# separate features and target variable
train_X = training[["petal_length","petal_width"]]
train_y = training["species"]

test_X = test[["petal_length","petal_width"]]
test_y = test["species"]

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# implementing decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train_X,train_y)

prediction = model.predict(test_X)
accuracy = metrics.accuracy_score(prediction,test_y)

print("The accuracy of the Decision Tree is:", accuracy)

# plotting decision regions
plot_decision_regions(np.array(train_X), np.array(train_y), clf=model)
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("Decision Regions for Training Set")
plt.show()

plot_decision_regions(np.array(test_X), np.array(test_y), clf=model)
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("Decision Regions for Test Set")
plt.show()

# plotting decision tree
tree.plot_tree(model,filled=True)
plt.show()