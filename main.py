import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz

#Open file with dataset and parse for heading and data
data = []
headings = []
with open('training_data.csv', 'r') as f:
  for i, line in enumerate(f):
    line = line.rstrip('\n') #Remove newline character
    #Check if it is the first line or not
    if i==0:
      headings = line.split(',')
    else:
      data.append(line.split(',')) #Create sub lists

# Create numpy array with training data
dataset = np.array(data)

#Create pandas data frame
df2 = pd.DataFrame(data, columns=headings)
blankIndex = [''] * len(df2)
df2.index = blankIndex
#print(df2)

#Create feature vectors
X = df2.drop('willwait', axis=1)
y = df2['willwait']

#Use one hot encoding for the categorical features
ohe = preprocessing.OneHotEncoder()
X_encoded = ohe.fit_transform(X)
#X_encoded = pd.get_dummies(X)

#Encode strings to numerical features for the target variable
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

#Decision tree classifier with the entropy option
#Uses Shannon entropy equation, shown here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_encoded, y_encoded)

#Predict new values
new_data = [['no', 'no', 'no', 'yes', 'full', '$', 'no', 'yes', 'french', '0-10']]
#Encode new data set
new_data_df = pd.DataFrame(new_data, columns=X.columns)
new_data_encoded = ohe.transform(new_data_df)
#Predict and transform back to string
y_pred = dtc.predict(new_data_encoded)
print("Predicted output: ", le.inverse_transform(y_pred))

#Print the data tree
tree.plot_tree(dtc)
dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=ohe.get_feature_names_out(X.columns),
                                class_names=le.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("mytree1")

