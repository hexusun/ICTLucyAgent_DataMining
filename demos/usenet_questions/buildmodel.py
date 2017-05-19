"""
Copyright (c) 2015. Docentron PTY LTD. All rights reserved.
This material may not be reproduced, displayed, modified or distributed without
the express prior written permission of the copyright holder.

Demo of building a classifier from a feature data set
"""
__author__ = 'Docentron PTY LTD'


# train a classifier
import numpy as np
f = open("data/data_questions_feature.csv") # read training data
f.readline()  # skip header
data = np.loadtxt(f,delimiter=',')
y = data[:, 0]   # class label
x = data[:, 1:]  # features
n = len(y)

# train a svm classifier
from sklearn import svm
print "Building classification model"
C = 100
d = 3.0 # degree
g = 0.0 # gamma
r = 0.0 # coef0
kernel = 'linear'
probability = True
clf = svm.SVC(C=C, kernel=kernel, probability=probability) # , degree=d, gamma=g, coef0=r
r = clf.fit(x,y)
print r

# save the classification model so we can load it and use it for prediction later
import pickle
s = pickle.dump(clf, open("data_questions_model.p", "wb"))
