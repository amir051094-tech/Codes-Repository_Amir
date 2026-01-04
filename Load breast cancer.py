import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib_inline as plt
import matplotlib.pyplot as plt1
from sklearn.tree import plot_tree

data=load_breast_cancer()
dataset=pd.DataFrame(data=data['data'],columns=data['feature_names'])
print("data is ", data['feature_names'])

cancer=data.data
cancer2=data.target
print("cancer2 is " ,cancer2)

own_color=['yellow','green']
sns.scatterplot(data=dataset,x='mean radius',y='mean fractal dimension',hue=data['target'],palette=own_color)
own_color=['yellow','green']
plt1.show()
print("what is this ",data['data'])
from sklearn.model_selection import train_test_split
x=dataset.copy() #deep copy

y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
print("x_train_shape",x_train.shape,"y_train_shape",y_train.shape,"x_test_shape", x_test.shape,"y_test_shape",y_test.shape)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy')
m1=model.fit(x_train,y_train)
m1.get_params()
prediction=model.predict(x_test)
print(prediction)

from sklearn.metrics import accuracy_score
print("Accuracy score is ", round(100*accuracy_score(y_test,prediction),2),"%")

from sklearn.metrics import confusion_matrix
confusion_eval= confusion_matrix(y_test,prediction)
print ("confusion matrix is" ,confusion_eval)

plt1.figure(figsize=(20,10))

plot_tree(model,filled=True ,feature_names=data.feature_names,class_names=data.target_names )
plt1.show()

