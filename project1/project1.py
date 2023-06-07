import math
from collections import Counter
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv
import graphviz


def main():
    print("Hello, You will see the decision trees...\n")
    file_data_entropy = input("Please enter the attribute name for entropy:\n")

    # use this input to ask user
    input_data = list(map(str, input("Use a space to give different value. First enter Alternate value, Bar value, \n"
                                     "Fri/Sat value, Hungry value, Patrons value, Price value, Raining value, \n"
                                     "Reservation value and Type value: ").split()))

    # all the rows. array 2d
    dataArr = []

    # the second to the last rows. the data with the outputs. array 2d
    dataArr2 = []

    # for the first row. the attributes. array 1d
    columns = []

    with open("training_data.csv", 'r') as file:
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

    # file_data_entropy varialble
    entAnswer = df2[file_data_entropy]
    le = preprocessing.LabelEncoder()
    entAnswer = le.fit_transform(entAnswer)

    entropy_calculation(entAnswer)

    decision_tree_construction(df2, input_data)


def decision_tree_construction(data, inputs):
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
    print('\nData Tree is saved under "myTree2.pdf", raw values:')
    print(tree.plot_tree(dtc), end='\n')
    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=ohe.get_feature_names_out(X.columns),
                                    class_names=le.classes_,
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("./project1/mytree2")

    # Enter data into splitting function to evaluate performance
    splitting_criteria(dtc, X_encoded, y_encoded)

    # four parameter input_data array
    predictFromDataset(dtc, X, ohe, le, inputs)


def entropy_calculation(instances):
    # Count the occurrences of each class label
    counter = Counter(instances)

    # Calculate the probabilities of each class label
    total_instances = len(instances)
    probabilities = [count / total_instances for count in counter.values()]

    # Calculate the entropy using the entropy formula
    entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)
    print("\nEntropy is: " + str(entropy), end='\n')

    return entropy


def splitting_criteria(dtc, X, y):
    # Split the test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=0)
    y_pred = dtc.predict(X_test)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return


def classification(x, y):
    # the criterion take care of the entropy
    # the splitter parameter take care of selecting the best
    # feature with the highest information gain
    # it does do the splitting criteria
    dtc = tree.DecisionTreeClassifier(criterion="entropy", splitter="best")
    dtc.fit(x, y)
    return dtc


# four parameter input_data array
def predictFromDataset(dtc, X, ohe, le, data):
    # Predict new values
    new_data = [data]
    # Encode new data set
    new_data_df = pd.DataFrame(new_data, columns=X.columns)
    new_data_encoded = ohe.transform(new_data_df)
    # Predict and transform back to string
    y_pred = dtc.predict(new_data_encoded)
    print("Predicted output: ", le.inverse_transform(y_pred))


# Run main function
if __name__ == "__main__":
    main()
