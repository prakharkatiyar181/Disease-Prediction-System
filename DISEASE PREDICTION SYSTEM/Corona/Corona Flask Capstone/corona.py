import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle

df = pd.read_csv('corona_tested_individuals_ver_0083.english.csv')
df = df[df.cough != "None"]
df = df[df.fever != "None"]
df = df.astype({'cough': int,'fever': int,'sore_throat': int,'head_ache': int, 'shortness_of_breath': int})
df['age_60_and_above'] = df['age_60_and_above'].map({'No':0, 'Yes':1})
df['Result'] = df['corona_result'].map({'negative':0, 'positive':1})
df2 =pd.get_dummies(df['test_indication'])
df = pd.concat([df, df2],axis=1)
df.dropna(inplace=True)
X=df.drop(['Result','corona_result','gender','test_indication','Other','test_date'],axis=1).values
y=df["Result"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
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

# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, y_train)
# pickle.dump(knn,open('model6.pkl','wb'))
# model6=pickle.load(open('model6.pkl','rb'))