import pandas as pd
from numpy import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib
from sklearn import preprocessing


# show model
mapping_data = pd.read_csv('dynamicmappings.csv')
data = mapping_data['keyinname']
values = array(data)
label_encoder = preprocessing.LabelEncoder()
mapping_data['keyinLabel'] = label_encoder.fit_transform(values)
# mapping_data['keyinLabelValue'] = label_encoder.inverse_transform(mapping_data['keyinLabel'] )

# print(mapping_data)

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
print('accuracy of predicition :', score * 10 , 'of 10')

# prediction based on model
# model = DecisionTreeClassifier()
# model.fit(X.values, y.values)
# predictions = model.predict([[7,5],[2,8]])
# print(predictions)

# train one time and use always  with joblib
# joblib.dump(model,'mapping-recommender.joblib')
model = joblib.load('mapping-recommender.joblib')
# predict [ type =7(loop) , keyinLabel =5(Items as label code)]

# prediction Key Out  property name
predictions = model.predict([[7,5]])
print('KeyOut will be :',predictions[0])

# encoded to find keyIn property name
keyin_name = label_encoder.inverse_transform([5])
print('KeyIn will be :',keyin_name[0])
# [7,5] means  type = 7   keyinLabel = 5 so keyout will be Items , keyin will be items


# drow predicition tree
#
# tree.export_graphviz(model,out_file='mapping-recommender.dot',
#                      feature_names=['keyinname','type'],
#                      class_names=sorted(y.unique()),
#                      label='all',
#                      rounded=True,
#                      filled=True)


# use this web site to drow diagram of ML algoritm https://dreampuf.github.io/GraphvizOnline/  or https://edotor.net/

# test
# valuestest = array(['Items','anna','ann','a','aa','ac','aaaw','asab','weww','sdasdsa','asdasdas','asasaeeww'])
# testkeyin = label_encoder.fit_transform(valuestest)
# print(testkeyin)