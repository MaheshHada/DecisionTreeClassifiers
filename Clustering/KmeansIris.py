#This code implements the KMeans on iris datset from scikit-learn 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

iris = datasets.load_iris()
iris.data
iris.feature_names
iris.target
iris.target_names

#storing data as Pandas Dataframes
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']


plt.figure(figsize=(14,7))
colormap = np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')

#KMeans model
model = KMeans(n_clusters=3)
model.fit(x)

#Results comparison between original and predicted values
plt.figure(figsize=(14,7))
colormap = np.array(['red','lime','black'])

#original
plt.subplot(1,2,1)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real Classification')

#prediction
plt.subplot(1,2,2)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[model.labels_],s=40)
plt.title('K Mean Classification')

predY = model.labels_
print(predY) #numpy array
print(y) #dataframe

#performance metrics
sm.accuracy_score(y,predY)

#confuson Matrix tells about the number of 
#correctly and mis classfied  classes
sm.confusion_matrix(y,predY)
#Refrence http://stamfordresearch.com/k-means-clustering-in-python/, thanks a lot