import math
from collections import Counter
import numpy as np
import scipy
from scipy.stats import entropy as en
from sklearn import tree
from sklearn import preprocessing
from IPython.display import display, HTML
import pandas as pd
import csv

# pd.set_option('display.max_columns', None)
# print("You will see the decision trees...\n")
# file_data = input("Please enter the file name:\n")

# all the rows
dataArr = []

# the second to the last rows
dataArr2 = []

# for the first row
columns = []

with open("./training_data.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        dataArr.append(row)


for row in dataArr[1:]:
    dataArr2.append(row)


dataset = np.array(dataArr2)
#print(dataset)

# first row
for element in dataArr[0]:
    columns.append(element)

df2 = pd.DataFrame(dataset.transpose(), columns)
blankIndex = [''] * len(df2)
df2.index = blankIndex

# display(HTML(df2.to_html()))
#print(df2)


def decision_tree_construction(data):
    X = data[:, 0:10]
    y = data[:, 10]
    le = preprocessing.LabelEncoder()
    X[:, 0:10] = le.fit_transform(X[:, 0:10])
    y = le.fit_transform(y)


def entropy_calculation(instances):
    # Count the occurrences of each class label
    counter = Counter(instances)

    # Calculate the probabilities of each class label
    total_instances = len(instances)
    probabilities = [count / total_instances for count in counter.values()]

    # Calculate the entropy using the entropy formula
    entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)

    return entropy


def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()           # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


def classification(x, y):
    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x, y)



