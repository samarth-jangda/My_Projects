#Firstly we will import the dataset usinf libraries
import pandas as pd
import matplotlib.pyplot as plt
from numpy import nan
import numpy as np
import seaborn as sns
df = pd.read_csv("C:\\Users\\NITIZEN\\Desktop\\Datasets\\churn_ibm.csv")
df.replace(nan,int(0))
print(df)
#then divide the depndent and independent varibles:
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)
fig1 = sns.violinplot(data = df, y = "MonthlyCharges" , x = "Churn")
fig2 = sns.scatterplot(data=df, y = "TotalCharges" , x = "MonthlyCharges")
plt.grid()
plt.show()
print(X.dtypes)
print("hello")
            # by converting columns from categorical to o and 1.
for column in X.columns:
    if X[column].dtype == np.object:
        print('Converting', column)   # converting the values
        X = pd.concat([X,pd.get_dummies(X[column],prefix = column,drop_first=True)],axis=1).drop([column],axis=1)  #assigning 0 and 1

print("hello")
y = pd.get_dummies(y,prefix='churn',drop_first=True)  #this is dependent variable which is also categorical hence converting it to 0 and 1
y.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=3)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train,y_train)
prediction = decision_tree.predict(X_test)
print("Accuracy",accuracy_score(y_test,prediction))
print("AUC",roc_auc_score(y_test,prediction))
print("Number of nodes",decision_tree.tree_.node_count)
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_test,prediction)
plt.plot(fpr,tpr,lw = 1, alpha=0.3,label ="ROC",color = "black")
plt.title("Roc Curve")
plt.grid()
plt.show()
print("hello")
     # lets save tree from overfitting hence we prepare new tree
decision_tree2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, criterion='entropy')
decision_tree2.fit(X_train,y_train)
prediction = decision_tree2.predict(X_test)
print("Accuracy",accuracy_score(y_test,prediction))
print("AUC", roc_auc_score(y_test,prediction))
print("Number Of Nodes", decision_tree2.tree_.node_count)
  # lets visualize the tree
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
from graphviz import Source
from sklearn import tree
Source(tree.export_graphviz(decision_tree2,out_file=None,feature_names=X.columns))
from IPython.display import SVG
graph = Source(tree.export_graphviz(decision_tree2,out_file=None,feature_names=X.columns))
SVG(graph.pipe(format= 'svg'))
graph = Source(tree.export_graphviz(decision_tree2,out_file=None,feature_names=X.columns))
graph.format = 'png'
graph.render('decision_tree2_render', view=True)
graph = Source(tree.export_graphviz(decision_tree2,out_file=None,feature_names=X.columns))
png_bytes = graph.pipe(format = 'png')
with open('decision_tree2.png', 'wb') as f:
    f.write(png_bytes)

from IPython.display import Image
Image(png_bytes)