from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
knn = KNeighborsClassifier(3).fit(X_tr, y_tr)
pred = knn.predict(X_te)
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))

print('--------------------------------------------------------')

import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('clean.csv')
feats = ['pclass', 'age', 'fare', 'family', 'sex_num']
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df['survived'], test_size=.25, random_state=0)
lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print('accuracy:', lr.score(X_te, y_te), 'baseline:', (y_te == 0).mean())

print('--------------------------------------------------------')

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_te, lr.predict(X_te)))
print(classification_report(y_te, lr.predict(X_te)))

print('--------------------------------------------------------')

new = pd.DataFrame([[1, 25, 80, 0, 1]], columns=feats)
print(lr.predict(new), lr.predict_proba(new))

print('--------------------------------------------------------')
print('ДОМАШНЯЯ РАБОТА')

print('k = 1')

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
knn = KNeighborsClassifier(1).fit(X_tr, y_tr)
pred = knn.predict(X_te)
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))

print('--------------------------------------------------------')
print('k = 5')

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
knn = KNeighborsClassifier(5).fit(X_tr, y_tr)
pred = knn.predict(X_te)
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))

print('--------------------------------------------------------')
print('k = 15')

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
knn = KNeighborsClassifier(15).fit(X_tr, y_tr)
pred = knn.predict(X_te)
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))