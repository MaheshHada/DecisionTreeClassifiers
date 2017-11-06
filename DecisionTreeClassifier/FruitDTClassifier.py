import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn import tree
from sklearn.datasets import load_iris

df = pd.DataFrame()
#generating data
df["smooth"] = np.array([9,10,8,8,7,3,4,2,5,6])
df["weight"] = np.array([170,175,180,178,182,130,120,130,138,145])
df["fruit"] = np.array([1,1,1,1,1,0,0,0,0,0])

#model 
clf = tree.DecisionTreeClassifier(criterion = "entropy")

#fitting the model on data
clf.fit(df[["smooth","weight"]],df["fruit"])

#obtaining the decision tree
with open("fruit_classifier.dot", "w") as f:
    f = tree.export_graphviz(clf, out_file = f)
#dot -Tpdf fruit_classifier.dot -o fruit_clf.pdf

#pair plot for the features
import seaborn as sns
%matplotlib inline
sns.set(style="ticks", color_codes = True)
g = sns.pairplot(data=df,y_vars = ["smooth","weight"],x_vars = ["smooth","weight"],hue = "fruit")

#checking the real and classified values
#Testing the model
test_features_1 = [[df["weight"][0],df["smooth"][0]]]
test_features_1_fruit = clf.predict(test_features_1)

test_features_3 = [df["weight"][2],df["smooth"][2]]
test_features_3_fruit = clf.predict(test_features_3)

test_features_8 = [df["weight"][7],df["smooth"][7]]
test_features_8_fruit = clf.predict(test_features_8)

print("Actual Fruit type = {act_fruit} , Fruit classifier predicted = {pred_fruit}".
      format(act_fruit = df["fruit"][0],pred_fruit = test_features_1_fruit))

print("Actual Fruit type = {act_fruit} , Fruit classifier predicted = {pred_fruit}".
      format(act_fruit = df["fruit"][2],pred_fruit = test_features_3_fruit))

print("Actual Fruit type = {act_fruit}, Fruit classfier predicted = {pred_fruit}".
      format(act_fruit = df["fruit"][7],pred_fruit = test_features_8_fruit))

#labelling the classes
fruit_label = ["orange","apple"]
f_label = []
for i in range(len(df)):
        if df["fruit"][i] == 0:
            f_label.append(fruit_label[0])
        else:
            f_label.append(fruit_label[1])
df['label'] = f_label
 
#printing the class with class labels
for i in range(len(df)):
    print (df["fruit"][i], df["smooth"][i],
           df["weight"][i],df["label"][i])

clf.score([[9,170],[6,145]],[[1],[0]])
clf.score(df[["smooth","weight"]],df["fruit"])