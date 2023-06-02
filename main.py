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
  # all the rows
  dataArr = []
  # the second to the last rows
  dataArr2 = []
  # for the first row
  columns = []

  # Open and read CSV file
  with open("./training_data.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
          dataArr.append(row)

  for row in dataArr[1:]:
      dataArr2.append(row)

  # Create numpy array with training data
  dataset = np.array(dataArr2)

  # first row
  for element in dataArr[0]:
      columns.append(element)

  #Create pandas data frame
  df2 = pd.DataFrame(dataArr2, columns=columns)
  blankIndex = [''] * len(df2)
  df2.index = blankIndex
  #print(df2)

  decision_tree_construction(df2)


def decision_tree_construction(data):
    #Create feature vectors
    X = data.drop('willwait', axis=1)
    y = data['willwait']

    #Use one hot encoding for the categorical features
    ohe = preprocessing.OneHotEncoder()
    X_encoded = ohe.fit_transform(X)

    #Encode strings to numerical features for the target variable
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)

    dtc = classification(X_encoded, y_encoded)

    #Print the data tree
    tree.plot_tree(dtc)
    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=ohe.get_feature_names_out(X.columns),
                                    class_names=le.classes_,
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("mytree1")


def classification(X, y):
    #Decision tree classifier with the entropy option
    #Uses Shannon entropy equation, shown here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y)
    return dtc


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


def predictFromDataset():
  #Predict new values
  new_data = [['no', 'no', 'no', 'yes', 'full', '$', 'no', 'yes', 'french', '0-10']]
  #Encode new data set
  new_data_df = pd.DataFrame(new_data, columns=X.columns)
  new_data_encoded = ohe.transform(new_data_df)
  #Predict and transform back to string
  y_pred = dtc.predict(new_data_encoded)
  print("Predicted output: ", le.inverse_transform(y_pred))


#Run main function
if __name__=="__main__":
   main()

