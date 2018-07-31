# ML-MODELS
models repository
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
df = pd.read_csv("titanic_train.csv")
df.head()
df.info()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop('Cabin',axis=1,inplace=True)
sns.boxplot(x='Age',y='Pclass',data=df,palette='rainbow')
def impute_Age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 21
        elif Pclass == 2:
            return 22
        else:
            return 23
    else:
        return Age
df['Age']=df[['Age','Pclass']].apply(impute_Age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.head()
df.info()
Sex = pd.get_dummies(df['Sex'],drop_first=False)
Embarked = pd.get_dummies(df['Embarked'],drop_first=False)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df = pd.concat([df,'Sex','Embarked'],axis=1)
df.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop('Survived',axis=1),df['Survived'],test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
pred = logmodel.predict(x_test)
print(classification_report(y_test,pred))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)
pred = svm.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(x_train,y_train)
pred = grid.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train,y_train)
pred = dtree.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
pred = rfc.predict(x_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
