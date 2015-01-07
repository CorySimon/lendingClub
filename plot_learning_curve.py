import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import re #regex

##Load Data
files = ['svc_learning_curve_round4_1_20140105.txt',
		 'svc_learning_curve_round4_2_20140105.txt',
		 'svc_learning_curve_round4_3_20140105.txt']

fpr_train = dict()
fpr_validate = dict()

num_regex = re.compile('\d+\.?\d+')
num_samples_regex = re.compile('Training classifier on')
train_regex = re.compile('Training Scores:')
validate_regex = re.compile('Cross Validation Scores:')
fpr_regex = re.compile('fpr:')


##Strip the number of samples and fpr's out of the log files
for fileName in files:
	print "Opening %s" %fileName
	f = open(fileName, 'r')
	num_samples = 0
	line = f.readline()
	while line:
		num_samples_regex_matches = num_samples_regex.findall(line)
		train_regex_matches = train_regex.findall(line)
		validate_regex_matches = validate_regex.findall(line)
		if num_samples_regex_matches:
			num_samples = num_regex.findall(line)
			num_samples = int(num_samples[0])
			line = f.readline()
		elif train_regex_matches:
			line = f.readline()
			fpr_regex_match = fpr_regex.findall(line)
			while not fpr_regex_match:
				line = f.readline()
				fpr_regex_match = fpr_regex.findall(line)
			_fpr_train = num_regex.findall(line)
			_fpr_train = float(_fpr_train[0])
			#store these fpr values in a dictionary
			if num_samples in fpr_train:
				fpr_train[num_samples].append(_fpr_train)
			else:
				fpr_train[num_samples] = [_fpr_train]

			line = f.readline()
		elif validate_regex_matches:
			line = f.readline()
			fpr_regex_match = fpr_regex.findall(line)
			while not fpr_regex_match:
				line = f.readline()
				fpr_regex_match = fpr_regex.findall(line)
			_fpr_validate = num_regex.findall(line)
			try:
				_fpr_validate = float(_fpr_validate[0])
			except IndexError:
				#this happens when fpr = nan
				_fpr_validate = 0.0
			#store these fpr values in a dictionary
			if num_samples in fpr_validate:
				fpr_validate[num_samples].append(_fpr_validate)
			else:
				fpr_validate[num_samples] = [_fpr_validate]
			line = f.readline()
		else:
			line = f.readline()
	f.close()

num_samples_vals = []
fpr_train_vals = []
fpr_train_min_vals = []
fpr_train_max_vals = []
fpr_validate_vals = []
fpr_validate_min_vals = []
fpr_validate_max_vals = []

iterator = iter(sorted(fpr_train.items()))
training_data = iterator.next()
while training_data:
	num_samples_vals.append(training_data[0])
	fpr_train_sum = 0
	##keep track of min and max to plot the spread
	fpr_train_min = training_data[1][0]
	fpr_train_max = training_data[1][0]
	for item in training_data[1]:
		fpr_train_sum += item
		if item < fpr_train_min:
			fpr_train_min = item
		if item > fpr_train_max:
			fpr_train_max = item
	fpr_train_mean = fpr_train_sum / len(training_data[1])
	fpr_train_vals.append(fpr_train_mean)
	fpr_train_min_vals.append(fpr_train_min)
	fpr_train_max_vals.append(fpr_train_max)
	try:
		training_data = iterator.next()
	except StopIteration:
		break

iterator = iter(sorted(fpr_validate.items()))
validate_data = iterator.next()
while validate_data:
	fpr_validate_sum = 0
	##keep track of min and max to plot the spread
	fpr_validate_min = validate_data[1][0]
	fpr_validate_max = validate_data[1][0]
	for item in validate_data[1]:
		fpr_validate_sum += item
		if item < fpr_validate_min:
			fpr_validate_min = item
		if item > fpr_validate_max:
			fpr_validate_max = item
	fpr_validate_mean = fpr_validate_sum / len(validate_data[1])
	fpr_validate_vals.append(fpr_validate_mean)
	fpr_validate_min_vals.append(fpr_validate_min)
	fpr_validate_max_vals.append(fpr_validate_max)
	try:
		validate_data = iterator.next()
	except StopIteration:
		break

print len(num_samples_vals)
print len(fpr_validate_vals)

plt.plot(num_samples_vals, fpr_train_vals, 'blue', label='Training', linewidth=3)
plt.fill_between(num_samples_vals, fpr_train_min_vals, fpr_train_max_vals, edgecolor='none', facecolor='blue', alpha=0.3)

plt.plot(num_samples_vals, fpr_validate_vals, 'red', label='Validation', linewidth=3)
plt.fill_between(num_samples_vals, fpr_validate_min_vals, fpr_validate_max_vals, edgecolor='none', facecolor='red', alpha=0.3)

plt.title('SVC Learning Curve - Round 4')
plt.text(1E-2, 1.0, 'FPR => Lower is Better')
plt.xlabel('Number of Samples')
plt.ylabel('False Positive Rate')
plt.ylim([0, 1.1])
plt.legend(loc='best')
plt.show()