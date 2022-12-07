import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np

test = pd.read_csv('testing.csv')
train = pd.read_csv('training.csv')
train.dropna(inplace=True,axis=1)
X_train = train.drop(['prognosis'], axis = 1)
y_train = train['prognosis']
X_test = test.drop(['prognosis'], axis = 1)
y_test = test['prognosis']

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pickle.dump(rfc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

lr = LogisticRegression()
lr.fit(X_train,y_train)
pickle.dump(lr,open('model2.pkl','wb'))
model2=pickle.load(open('model2.pkl','rb'))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pickle.dump(dt,open('model3.pkl','wb'))
model3=pickle.load(open('model3.pkl','rb'))

nb = GaussianNB()
nb.fit(X_train,y_train)
pickle.dump(nb,open('model4.pkl','wb'))
model4=pickle.load(open('model4.pkl','rb'))


mlp = MLPClassifier(max_iter=100)
mlp.fit(X_train,y_train)
pickle.dump(mlp,open('model5.pkl','wb'))
model5=pickle.load(open('model5.pkl','rb'))

svc = SVC()
svc.fit(X_train,y_train)
pickle.dump(svc,open('model6.pkl','wb'))
model5=pickle.load(open('model6.pkl','rb'))

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pickle.dump(knn,open('model7.pkl','wb'))
model6=pickle.load(open('model7.pkl','rb'))