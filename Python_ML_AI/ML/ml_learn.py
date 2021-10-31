import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# train ML and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)

# prediction based on model
model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([[20,1]])
print(predictions)


# predicition with model

# joblib.dump(model,'music-recommender.joblib')
model = joblib.load('music-recommender.joblib')
predictions = model.predict([ [32, 0], [30, 1]])
print(predictions)

# drow predicition tree

tree.export_graphviz(model,out_file='music-recommender.dot',
                     feature_names=['age','gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)