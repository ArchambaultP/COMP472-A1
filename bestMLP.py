from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd

def bestMLP(X_train, Y_train, X_val, Y_val, X_test):
	
	#creates the model with the specified paramters (used for testing different parameters) and returns the model
	def model(hlsizes, a, s):
		mlp = MLPClassifier(hidden_layer_sizes = hlsizes, activation = a, solver = s, random_state=1, max_iter=200)
		mlp.fit(X_train, Y_train)
		return mlp		

	#outputs preditions of a given model and a given dataset
	def predictions(model, X):
		pred = model.predict(X)
		return pred

	#used to compute stats to compare different models (avg of f1macro f1micro and accuracy)
	def stats(pred):
		f1Micro = f1_score(Y_val, pred, average='micro')
		f1Macro = f1_score(Y_val, pred, average='macro')
		accuracy = accuracy_score(Y_val, pred)
		return (f1Micro + f1Macro + accuracy) / 3

	#this function is used to find the best set of parameters given a parameter list (grid search) and outputs the best parameters 
	def bestParams():
		best = 0
		bestparams = [] 
		parameters = {
    			'hidden_layer_sizes': [(30,50,), (10,10,10,)],
    			'activation': ['tanh', 'relu', 'logistic', 'identity'],
    			'solver': ['sgd', 'adam'],
    			}
		for x in parameters['hidden_layer_sizes']:
			for y in parameters['activation']:
				for z in parameters['solver']:
					pred = predictions(model(x,y,z), X_val)
					print(x,y,z)
					statss = stats(pred)
					print(statss)
					if statss > best:
						best = statss
						bestparams = [x,y,z]
		return bestparams
	
	#using the best params for the output of this model
	return predictions(model((30,50), 'tanh', 'adam'), X_test)


#here are my results for looking for the best parameters 

# best = bestParams()
# print(predictions(model(best[0], best[1], best[2]), X_test))
'''
(30, 50) tanh sgd
0.7089382862033888

#BEST
(30, 50) tanh adam
0.8239720242121354

(30, 50) relu sgd
0.7263708338421706
(30, 50) relu adam
0.7346013017638647
(30, 50) logistic sgd
0.033260074762659055
(30, 50) logistic adam
0.785599070378197
(30, 50) identity sgd
0.7976553984795393
(30, 50) identity adam
0.8185398856734106
(10, 10, 10) tanh sgd
0.146742865594686
(10, 10, 10) tanh adam
0.4640217931107473
(10, 10, 10) relu sgd
0.3350647564327201
(10, 10, 10) relu adam
0.4697632685314767
(10, 10, 10) logistic sgd
0.02892376285530492
(10, 10, 10) logistic adam
0.09005681701123235
(10, 10, 10) identity sgd
0.5966698659093936
(10, 10, 10) identity adam
0.6739005597071898
'''

