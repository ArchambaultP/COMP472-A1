from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def bestMLP(X_train, Y_train, X_val, Y_val, X_test, Y_test):
	



	def model():
		
		parameters = {
    				'hidden_layer_sizes': [(30,50), (10,10,10)],
    				'activation': ['tanh', 'relu', 'sigmoid', 'identity'],
    				'solver': ['sgd', 'adam'],
    				}
   
		mlp = MLPClassifier()
		gsMLP = GridSearchCV(mlp, parameters)
		gsMLP.fit(X_train, Y_train)

		print(gsMLP.predict(X_val))
		

	def predictions(model, X_test, Y_test):
		
		pred = model.predict(X_test)
		return pred

	model()


