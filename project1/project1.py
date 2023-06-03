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
import graphviz


def main():
    # pd.set_option('display.max_columns', None)
    print("Hello, You will see the decision trees...\n")
    # file_data = input("Please enter the file name:\n")
    input_data = list(map(str, input("Use a space to give different value. First enter Alternate value, Bar value, \n"
                                     "Fri/Sat value, Hungry value, Patrons value, Price value, Raining value, \n"
                                     "Reservation value and Type value: ").split()))
    print(input_data)

    # all the rows. array 2d
    dataArr = []

    # the second to the last rows. the data with the outputs. array 2d
    dataArr2 = []

    # for the first row. the attributes. array 1d
    columns = []

    with open("./training_data.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            dataArr.append(row)

    for row in dataArr[1:]:
        dataArr2.append(row)

    # first row
    for element in dataArr[0]:
        columns.append(element)

    df2 = pd.DataFrame(dataArr2, columns=columns)
    blankIndex = [''] * len(df2)
    df2.index = blankIndex

    print(df2, end='\n')

    decision_tree_construction(df2, columns)


def decision_tree_construction(data, columns):
    # Create feature vectors
    X = data.drop('willwait', axis=1)
    y = data['willwait']

    # Use one hot encoding for the categorical features
    ohe = preprocessing.OneHotEncoder()
    X_encoded = ohe.fit_transform(X)

    # Encode strings to numerical features for the target variable
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)

    dtc = classification(X_encoded, y_encoded)

    # Print the data tree
    print(tree.plot_tree(dtc), end='/n')
    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=ohe.get_feature_names_out(X.columns),
                                    class_names=le.classes_,
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("mytree2")

    #predictFromDataset(dtc, columns)


def entropy_calculation(instances):
    # Count the occurrences of each class label
    counter = Counter(instances)

    # Calculate the probabilities of each class label
    total_instances = len(instances)
    probabilities = [count / total_instances for count in counter.values()]

    # Calculate the entropy using the entropy formula
    entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)

    return entropy


def splitting_criteria():
    return


def classification(x, y):
    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x, y)
    return dtc

# having issue with this method
def predictFromDataset(dtc, arrCollumns):
    # Predict new values
    new_data = [['no', 'no', 'no', 'yes', 'full', '$', 'no', 'yes', 'french', '0-10']]

    # Encode strings to numerical features for the target variable
    le = preprocessing.LabelEncoder()
    new_data_encoded = le.fit_transform(new_data)
    #new_data_encoded.reshape(-1, 1)

    # Predict and transform back to string
    y_pred = dtc.predict(new_data_encoded)
    print("Predicted output: ", le.inverse_transform(y_pred))


# Run main function
if __name__ == "__main__":
    main()
