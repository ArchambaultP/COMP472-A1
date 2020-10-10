import csv
from sklearn.metrics import confusion_matrix
import numpy as np


def createCSV(pred, Y_test, symbolDict, outName):

	file = open(f'{outName}.csv', 'w')
	writer = csv.writer(file)
	writer.writerows(confusion_matrix(Y_test, pred))
