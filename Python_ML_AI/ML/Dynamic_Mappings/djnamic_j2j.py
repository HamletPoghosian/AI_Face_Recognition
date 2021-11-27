import pandas as pd
from numpy import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

mapping_data = pd.read_csv('dynamicmappings.csv')
X = mapping_data.drop(columns=['keyinname','keyout'])
y = mapping_data['keyout']

# encoding value to set into algortim , string cant use in ML , only Integer ,
# # train=convert(X)
# # define example
# data = mapping_data['keyin']
#
# values = array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)


# train ML and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train.values, y_train.values)
predictions = model.predict(X_test)
score = accuracy_score(y_test.values, predictions)
print(score)

# prediction based on model
model = DecisionTreeClassifier()
model.fit(X.values, y.values)
predictions = model.predict([[7,7],[5,7]])
print(predictions)


# joblib.dump(model,'mapping-recommender.joblib')
model = joblib.load('mapping-recommender.joblib')
predictions = model.predict([[7,7],[5,7]])
print(predictions)

# drow predicition tree

tree.export_graphviz(model,out_file='mapping-recommender.dot',
                     feature_names=['keyinname','type'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)


# use this web site to drow diagram of ML algoritm https://dreampuf.github.io/GraphvizOnline/