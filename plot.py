import matplotlib.pyplot as plt
import numpy as np
from numpy import random

##Put the series you want to plot in the double array data. Put the series labels in the labels array
##Right now, all y_data series must use the same x_series


x_data = [1E-4, 1E-3, 1E-2, 1E-1, 1E0]
y_data = [[0.0, 0.0, 0.339, 0.0, 0.0], [0.0, 0.001, 0.316, 0.0, 0.0],
		  [0.014, 0.383, 0.355, 0.147, 0.0], [0.013, 0.367, 0.312, 0.093, 0.0],
		  [0.407, 0.349, 0.376, 0.490, 0.996], [0.385, 0.333, 0.303, 0.105, 0.0]]
labels = ['C=0.01 Test', 'C=0.01 Train', 'C=0.1 Test', 'C=0.1 Train', 'C=0.5 Test', 'C=0.5 Train']
color = ['#0015ff', '#353c85', '#33FF00', '#439130', '#FF0000', '#9C2F2F', '#EA00FF', '#85348C']
for i in np.arange(0,len(y_data)):
	plt.semilogx(x_data, y_data[i], color[i], label=labels[i], linewidth=3)

plt.title('SVC Round 4')
plt.text(1E-2, 1.0, 'FPR => Lower is Better (0 is bad)')
plt.text(1E-2, 0.8, '19847 train, 19847 validation')
plt.xlabel('gamma')
plt.ylabel('False Positive Rate')
plt.ylim([0, 1.1])
plt.legend(loc='best')
plt.show()