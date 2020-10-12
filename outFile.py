import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np 


def createCSV(pred, Y_test, symbolDict, outName):
	
	#part 3 a)
	def rowPred():
		out =[]
		for x in range(len(pred)):
			out.append(f"{x},{pred[x]}")
		
		#we can also add a representation of the actual value eg. (0,1 	A) its not asked tho 
		dictA = {
			'Row,Prediction' : out
		}

		#returning a dataframe for part A
		return pd.DataFrame(dictA)

	#part 3 b)
	def confMatrix():

		confMatrix = confusion_matrix(Y_test, pred)
		#adding the key on top of the matrix so we have a visual representation of the values being compared
		keys = [k for k in symbolDict.keys()]
		confMatrix =  np.append([keys], confMatrix, axis=0)
		
		#adding the values on the left so we can see whats being compared
		key0 = [[k] for k in symbolDict.keys()]
		key0 = [[""]] + key0
		confMatrix =  np.append(key0, confMatrix, axis=1)

		#i needed to push the matrix to the right because other data was making some of the csv boxes bigger so it wasnt
		#very clear what was going on so i moved it out of the way, we can figure out other ways 
		for x in range(4):
			pushRight = [[""] for k in range(len(confMatrix))]
			confMatrix =  np.append(pushRight, confMatrix, axis=1)

		return confMatrix

	#When calling these functions I checked and they are automatically sorted (first value of the score corresponds to value 0(A)) 

	#part 3 c)	
	def statPerClass():

		percision =  precision_score(Y_test, pred, average = None)
		recall = recall_score(Y_test, pred, average=None)
		f1 = f1_score(Y_test, pred, average=None)

		dictC = {
			'Class' : [k for k in symbolDict.keys()],
			'Percision' : percision,
			'Recall' : recall,
			'F1' : f1
		}
		#returning a dataframe of the values needed for part C
		return pd.DataFrame(dictC).round(2)

	#part 3d)
	def avgStat():

		f1Micro = f1_score(Y_test, pred, average='micro')
		f1Macro = f1_score(Y_test, pred, average='macro')
		accuracy = accuracy_score(Y_test, pred)

		dictD = {
			'Accuracy' : [accuracy],
			'Micro-Average-F1' : [f1Micro],
			'Macro-Average-F1' : [f1Macro]
			}

		#returning dataframe with info for part D 
		return pd.DataFrame(dictD).round(4)



	partA = rowPred()
	partB = confMatrix()
	partD = avgStat()
	partC = statPerClass()


	#i did it like this so we can append dataframes one under another, i dont know if this is the best
	#way, we can discuss it later 

	with open(outName, 'w') as file:
		partA.to_csv(file, index = False)

	with open(outName, 'a') as file:
		writer = csv.writer(file)
		writer.writerows("\n")	
		writer.writerows(partB)
		writer.writerows("\n")	

	with open(outName, 'a') as file:
		partC.to_csv(file, index = False)

	with open(outName, 'a') as file:
 	   writer = csv.writer(file)
 	   writer.writerows("\n")		
 	   partD.to_csv(file, index = False)

