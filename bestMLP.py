from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
def bestMLP(X_train, Y_train, X_val, Y_val, X_test):
	

	def model(hlsizes, a, s):
		mlp = MLPClassifier(hidden_layer_sizes = hlsizes, activation = a, solver = s)
		mlp.fit(X_train, Y_train)
		return mlp		

	def predictions(model, X):
		pred = model.predict(X)
		return pred

	def stats(pred):
		f1Micro = f1_score(Y_val, pred, average='micro')
		f1Macro = f1_score(Y_val, pred, average='macro')
		accuracy = accuracy_score(Y_val, pred)
		return (f1Micro + f1Macro + accuracy) / 3

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


	best = bestParams()

	print(predictions(model(best[0], best[1], best[2]), X_test))
