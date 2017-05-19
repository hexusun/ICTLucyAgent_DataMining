"""
Copyright (c) 2015. Docentron PTY LTD. All rights reserved.
This material may not be reproduced, displayed, modified or distributed without
the express prior written permission of the copyright holder.

Demo of evaluating a SVM classifier on a data set
"""
__author__ = 'Docentron PTY LTD'

import sys

# train a classifier
import numpy as np
f = open("data/data_questions_feature.csv")
f.readline()  # skip header
data = np.loadtxt(f,delimiter=',')
y = data[:, 0]   # class label
x = data[:, 1:]  # features
n = len(y)

# cross validation
import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold

# SVM parameters
kernel = 'linear' # linear, poly (d,g,r), rbf (g), sigmoid (g,r)
C = 100
d = 3.0 # degree
g = 0.0 # gamma
r = 0.0 # coef0
probability = True

# Prepare KFold data sets
kf = KFold(n, n_folds=10)  # 10 fold cross validation. Usually 5 or 10 cross validation is good
y_target = np.array([])
y_predicted = np.array([])
y_score = None
for train, test in kf:
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    y_target = np.concatenate((y_target, y_test))

    clf = svm.SVC(C=C, kernel=kernel, probability=probability) # , degree=d, gamma=g, coef0=r
    r = clf.fit(x_train, y_train)
    y_p = clf.predict(x_test)
    y_s = clf.predict_proba(x_test)
    print y_s

    y_predicted = np.concatenate((y_predicted, y_p))
    if y_score is not None:
        y_score = np.concatenate((y_score, y_s))
    else:
        y_score = y_s

# Precision and recall score of the classifier
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_target, y_predicted, target_names=target_names))

# calc ROC for class 1
from sklearn.metrics import roc_curve, auc
fpr1, tpr1, thresholds1 = roc_curve(y_target, y_score[:,1], pos_label=1)
roc_auc1 = auc(fpr1, tpr1)

# calc ROC for class 0
fpr0, tpr0, thresholds = roc_curve(y_target, y_score[:,0], pos_label=0)
roc_auc0 = auc(fpr0, tpr0)

# Plot of the ROC curves for class 1
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr0, tpr0, label='ROC curve {0} (area = {1:0.2f})'.format(0, roc_auc0))
plt.plot(fpr1, tpr1, label='ROC curve {0} (area = {1:0.2f})'.format(1, roc_auc1))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot the micro-average ROC
# Plot the macro-average ROC
