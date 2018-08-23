
# Import the libraries
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = dataset.data 
labels = dataset.target



# Extract the hog features
list_hog_ft = []
for feature in features:
    ft = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_ft.append(ft)
hog_features = np.array(list_hog_ft, 'float64')



# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)



from sklearn import feature_extraction, model_selection, metrics, svm
X_train, X_test, y_train, y_test = model_selection.train_test_split(hog_features, labels, test_size=0.3)
print([np.shape(X_train), np.shape(X_test)])



# Experiment on different values of C
list_C = np.arange(4, 10, 1) #100000
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
count = 0
for C in list_C:
    svc = svm.SVC(C=C,probability=True)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    count = count + 1

import pandas as pd
matrix = np.matrix(np.c_[list_C, score_train, score_test])
models = pd.DataFrame(data = matrix, columns = 
             ['C', 'Train Accuracy', 'Test Accuracy'])
models.head()


best_index = models['Test Accuracy'].idxmax()
models.iloc[best_index, :]

best_index=11
svc = svm.SVC(C=list_C[best_index],probability=True)
svc.fit(X_train, y_train)
models.iloc[best_index, :]

m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# Create an SVM classifier
clf = SVC(C=5,gamma=.05)

# Perform the training
clf.fit(hog_features, labels)

clf.score(hog_features, labels)

# Save the classifier
joblib.dump((clf, pp), "multi_digits_svm.pkl", compress=3)

