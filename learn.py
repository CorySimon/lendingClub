import pandas as pd
import numpy as np
import glob, sys
from datetime import date
import pickle
from random import randint

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



##Load loanData
print "loading data..."
f = open('data.pickle', 'rb')
loanData = pickle.load(f)

##remove all non-finished loans
loanData = loanData[[x in ['Charged Off', 'Default', 'Fully Paid'] for x in loanData['loan_status']]]
loanData.index = range(len(loanData))

##remove all states
loanData = loanData.drop('addr_state', 1)
loanData = loanData.drop('total_pymnt', 1)
loanData = loanData.drop('pymnt_plan', 1)
loanData = loanData.drop('initial_list_status', 1)
loanData = loanData.drop('mths_since_last_major_derog', 1)
loanData = loanData.drop('policy_code', 1)


##Relabel all defaulted loans as charged off and number them
for i in range(len(loanData['loan_status'])):
	if loanData['loan_status'][i] == 'Default':
		loanData['loan_status'][i] = 0
	elif loanData['loan_status'][i] == 'Charged Off':
		loanData['loan_status'][i] = 0
	else:
		loanData['loan_status'][i] = 1



labels = loanData['loan_status'].values
feature_names = list(loanData.columns.values)
feature_names.remove('loan_status')


features = loanData[feature_names].values

##Split into training and testing values
print "splitting into training and testing data..."
test_indices = []
train_indices = range(len(features))
while len(test_indices) < 10000:
	r = randint(0, len(labels)-1)
	if r not in test_indices:
		test_indices.append(r)
		train_indices.remove(r)

test_X = features[test_indices]
test_y = labels[test_indices]

##balance the training data by including 5 copies of each defaulted loan
print "balancing default loans..."
for i in range(len(train_indices)):
	if labels[train_indices[i]] == 0:
		for j in range(1,5):
			#add in the index 4 more times
			train_indices.append(train_indices[i])
X = features[train_indices]
y = labels[train_indices]


print "number of loans in training set: " , len(y)
print "number of defaults: ", np.sum(y == 0)



###Random Forest Classifier
n = [1,2,3,4,5,10,20]
pred = dict()
pred_proba = dict()

print "Running Random Forest:"
for j in n:

	clf = rfc(n_estimators=j, oob_score = True)
	clf = clf.fit(X, y)
	print j
	print "oob_score: ", clf.oob_score_

	print clf.feature_importances_ * 100
	print feature_names

	pred[j] = clf.predict(test_X)
	pred_proba[j] = clf.predict_proba(test_X)

	predict_paid_actually_paid = np.sum((pred == 1) & (test_y == 1))*1.0 / np.sum(test_y == 1)
	predict_paid_actually_default = np.sum((pred == 1) & (test_y == 0))*1.0 / np.sum(test_y == 0)
	predict_default_actually_paid = np.sum((pred == 0) & (test_y == 1))*1.0 / np.sum(test_y == 1)
	predict_default_actually_default = np.sum((pred == 0) & (test_y == 0))*1.0 / np.sum(test_y == 0)

	print "predict_paid_actually_paid: " + str(predict_paid_actually_paid)
	print "predict_paid_actually_default: " + str(predict_paid_actually_default)
	print "predict_default_actually_paid: " + str(predict_default_actually_paid)
	print "predict_default_actually_default: " + str(predict_default_actually_default)

	print "total default rate: " , np.sum(test_y == 0)*1.0 / np.size(test_y)
	print "algorithm's default rate: " , np.sum((pred == 1) & (test_y == 0))*1.0 / np.sum(pred == 1)
	print "\n"


##ROC Curves
fpr = dict()
tpr = dict()
thresholds = dict()
for j in n:

	fpr[j], tpr[j], thresholds[j] = roc_curve(test_y, pred_proba[j][:, 1], pos_label=1)

	# Plot of a ROC curve
	#plt.figure()
	plt.plot(fpr[j], tpr[j], label='ROC curve for %s classifiers' %j)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()


