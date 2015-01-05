import pandas as pd
import numpy as np
import glob, sys
from datetime import date
import pickle
from random import randint

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from sklearn.decomposition import PCA

from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

from time import time
from operator import itemgetter


class Trainer():

	def __init__(self):
		print "Loading Trainer Class..."
		self.start_time = time()
		self.loadData('data.pickle')
		self.splitTrainTest(self.labels, self.features)

	def loadData(self, fileName):

		print "Loading Data..."
		f = open(fileName, 'rb')
		loanData = pickle.load(f)

		self.labels = loanData['loan_status'].values
		feature_names = list(loanData.columns.values)
		feature_names.remove('loan_status')
		self.features = loanData[feature_names].values
		print "Data loaded. There are ", len(feature_names), " dimensions"

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed


	def splitTrainTest(self, labels, features):

		##Split data into train and test sets, then further split training data into 
		##train and cross-validate sets

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, 
															labels, 
															test_size=0.001)
		self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_train, self.y_train, test_size=0.5)


		print "number of loans in training set: " , len(self.y_train)
		print "number of defaults in training set: ", np.sum(self.y_train == 0)
		print "number of loans in CV set: " , len(self.y_cv)
		print "number of defaults in CV set: ", np.sum(self.y_cv == 0)
		print "number of loans in test set: " , len(self.y_test)
		print "number of defaults in test set: ", np.sum(self.y_test == 0)

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed


	def defineRFC(self, n_estimators=10, criterion='gini', max_depth=None, 
				  min_samples_split=2, min_samples_leaf=1, max_features='auto', 
				  max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
				  random_state=None, verbose=0, min_density=None, 
				  compute_importances=None):

		print "Using a Random Forest Classifier ..."
		self.clf = rfc(n_estimators=n_estimators, 
					   criterion=criterion, 
					   max_depth=max_depth, 
				  	   min_samples_split=min_samples_split, 
				  	   min_samples_leaf=min_samples_leaf, 
				  	   max_features=max_features, 
				 	   max_leaf_nodes=max_leaf_nodes, 
				 	   bootstrap=bootstrap, 
				 	   oob_score=oob_score, 
				 	   n_jobs=n_jobs, 
				  	   random_state=random_state, 
				  	   verbose=verbose, 
				  	   min_density=min_density, 
				  	   compute_importances=compute_importances)


	def defineSVC(self, C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, 
				  probability=False, tol=0.01, cache_size=200, class_weight='auto', verbose=True, 
				  max_iter=-1, random_state=None):

		print "Using a Support Vector Machine Classifier ..."
		self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, 
				  probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, 
				  max_iter=max_iter, random_state=random_state)

		print self.clf.get_params()


	def trainCLF(self):
		print "Training the Classifier..."
		self.clf.fit(self.X_train, self.y_train)

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed


	def getScores(self, X, y):
		print "Running CLF ..."

		pred = self.clf.predict(X)

		#tpr == predict paid, actually paid rate (ie recall)
		tpr = np.sum((pred == 1) & (y == 1))*1.0 / np.sum(y == 1)
		#fpr == predict paid, actually default rate
		fpr = np.sum((pred == 1) & (y == 0))*1.0 / np.sum(y == 0)
		#fnr == predict default actually paid rate
		fnr = np.sum((pred == 0) & (y == 1))*1.0 / np.sum(y == 1)
		#tnr == predict default actually default rate (ie specificity)
		tnr = np.sum((pred == 0) & (y == 0))*1.0 / np.sum(y == 0)

		print "tpr - predict_paid_actually_paid: " + str(tpr)
		print "fpr - predict_paid_actually_default: " + str(fpr)
		print "fnr - predict_default_actually_paid: " + str(fnr)
		print "tnr - predict_default_actually_default: " + str(tnr)

		print "total default rate: " , np.sum(y == 0)*1.0 / np.size(y)
		print "algorithm's default rate: " , np.sum((pred == 1) & (y == 0))*1.0 / np.sum(pred == 1)
		print "\n"

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed

		return [tpr, fpr, fnr, tnr]


	def standardizeSamples(self):
		##0 mean, unit variance
		self.X_train = preprocessing.scale(self.X_train)
		self.X_test = preprocessing.scale(self.X_test)
		self.X_cv = preprocessing.scale(self.X_cv)

	def scaleSamplesToRange(self):
		##Samples lie in range between 0 and 1
		minMaxScaler = preprocessing.MinMaxScaler()
		self.X_train = minMaxScaler.fit_transform(self.X_train)
		self.X_test = minMaxScaler.fit_transform(self.X_test)
		self.X_cv = minMaxScaler.fit_transform(self.X_cv)

	def normalizeSamples(self):
		##Unit norm
		self.X_train = preprocessing.normalize(self.X_train, norm='l2')
		self.X_test = preprocessing.normalize(self.X_test, norm='l2')
		self.X_cv = preprocessing.normalize(self.X_cv, norm='l2')

	def getRandomSamples(self, num_samples, X, y):
		train_size = num_samples*1.0 / np.size(y)
		_X_test, X_subset, _y_test, y_subset = train_test_split(X, y, test_size=train_size)
		return [X_subset, y_subset]

	def computeLearningCurve(self, min_samples=10, max_samples=30000, step_size=1000):
		##Store values in arrays to plot at the end
		train_fpr = []
		cv_fpr = []
		num_samples = []

		for n in np.arange(min_samples, max_samples, step_size):
			training_data_subset = self.getRandomSamples(n, self.X_train, self.y_train)
			X_train = training_data_subset[0]
			y_train = training_data_subset[1]

			print "\n\n # of training examples: ", len(y_train)
			
			print "\nTraining classifier on ", n, " samples ... "
			self.clf.fit(X_train, y_train)
			print "Training Scores: "
			training_scores = self.getScores(X_train, y_train)
			train_fpr.append(training_scores[1])

			cv_data_subset = self.getRandomSamples(n, self.X_cv, self.y_cv)
			X_cv = cv_data_subset[0]
			y_cv = cv_data_subset[1]
			print "Cross Validation Scores: "
			cv_scores = self.getScores(X_cv, y_cv)
			cv_fpr.append(cv_scores[1])

			num_samples.append(n)

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed

		plt.plot(num_samples, train_fpr, 'red', label='train', linewidth=3)
		plt.plot(num_samples, cv_fpr, 'blue', label='validation', linewidth=3)
		plt.xlabel('# of Training Samples')
		plt.ylabel('score')
		plt.legend(loc='best')
		plt.show()
		

	def runPCA(self, n_components=None, copy=False, whiten=False):
		print "Running PCA Dimensionality Reduction with n_components = ", n_components
		self.pca = PCA(n_components=n_components, copy=copy, whiten=whiten)
		self.X_train = self.pca.fit_transform(self.X_train)
		print "Reduced data down to ", self.pca.n_components_, " dimensions: "
		print "Transforming test data ..."
		self.X_test = self.pca.transform(self.X_test)
		self.X_cv = self.pca.transform(self.X_cv)

		time_elapsed = time() - self.start_time
		print "Time Elapsed: ", time_elapsed

	def runGridSearch(self):

		param_dist = {"max_depth": [3, None],
					  "max_features": sp_randint(1,11),
					  "min_samples_split": sp_randint(1,11),
					  "min_samples_leaf": sp_randint(1,11),
					  "bootstrap": [True, False],
					  "n_estimators": [10,20,30,50]}
		n_iter_search = 30

		randomSearch = RandomizedSearchCV(self.clf, param_distributions=param_dist, n_iter=n_iter_search, verbose=6, n_jobs=-1)
		start = time()
		randomSearch.fit(self.X_train, self.y_train)
		
		self.report(randomSearch.grid_scores_, 5)

		print "RandomizedSearchCV took %0.2f s for %d candidates" %((time()-start), n_iter_search)


	def runSVCGridSearch(self):
		C_vals = [0.001, 0.01, 0.1, 0.5]
		gamma_vals = [1E-4, 1E-3, 1E-2, 1E-1, 1]

		for C in C_vals:
			for gamma in gamma_vals:
				print "\n\n C: ", C, "  gamma: ", gamma
				self.defineSVC(C=C, gamma=gamma)
				self.trainCLF()
				print "Training Scores:"
				self.getScores(self.X_train, self.y_train)
				print "Testing Scores:"
				self.getScores(self.X_cv, self.y_cv)


	def computeROC(self):
		##ROC Curves
		fpr = dict()
		tpr = dict()
		thresholds = dict()
		pred_proba = dict()
		n = [100]
		
		for j in n:

			pred_proba[j] = self.clf.predict_proba(self.X_cv)
			fpr[j], tpr[j], thresholds[j] = roc_curve(self.y_cv, pred_proba[j][:, 1], pos_label=1)

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


	def report(self, grid_scores, n_top=2):
		top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
		for i, score in enumerate(top_scores):
			print "Model with Rank: {%i}" %(i+1)
			print "Mean Validation Score: %0.3f, std: %0.3f" %(score.mean_validation_score, np.std(score.cv_validation_scores))
			print "Parameters: ", score.parameters
			print ""


	def plot_with_err(self, x, data, color, **kwargs):
	    mu, std = data.mean(1), data.std(1)
	    plt.plot(x, mu, '-', c=color, **kwargs)
	    plt.fill_between(x, mu - std, mu + std, edgecolor='none', facecolor=color, alpha=0.2)
	    plt.xlabel('# of Training Samples')
	    plt.ylabel('score')
	    plt.legend(loc='best')
	    plt.show()


trainer = Trainer()
trainer.defineSVC(C=0.1, gamma=0.1)
trainer.scaleSamplesToRange()
trainer.standardizeSamples()
trainer.runPCA(n_components=50)
#trainer.runSVCGridSearch()
trainer.computeLearningCurve()

#trainer.trainCLF()
#print "Training Scores:"
#trainer.getScores(trainer.X_train, trainer.y_train)
#print "Testing Scores:"
#trainer.getScores(trainer.X_test, trainer.y_test)
#trainer.computeLearningCurve(1, 10000, 100)
#trainer.runGridSearch()

#runGridSearch()
#computeLearningCurve()
#computeROC()


