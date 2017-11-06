#This code implements KNN on MNIST dataset
from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2

mnist = datasets.load_digits()

#splitting the data in train_data,val_data,test_data

(trainData, testData, trainLabels, testLabels) = train_test_split(
np.array(mnist.data), mnist.target, test_size = 0.25, random_state = 42)

(trainData, valData, trainLabels, valLabels) = train_test_split(
trainData, trainLabels,test_size = 0.1, random_state = 84)

print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

#Hyperparameter Tuning: the best value of k
kVals = range(1, 30, 2)
accuracies = []

for k in range(1,30,2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)
    
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

#modelling the actual model with best value for the k (no of neighbors to consider) acquired.
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
print("Results of Classification")
print(classification_report(testLabels, predictions))


high =len(testLabels)
for i in np.random.randint(0, high, size = (5,)):
    image = testData[i]
    prediction = model.predict(image)[0]
    
    image = image.reshape((8,8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
    
    print("The digit is : {}",format(prediction))
    cv2.imshow("Image",image)
    cv2.waitKey(0)

#Refrence https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/#, thanks a lot