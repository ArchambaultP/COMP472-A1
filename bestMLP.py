from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def bestMLP(X_train, Y_train, X_val, Y_val, X_test, Y_test):
	
	def model(solverType, activationType, layers):
		
		mlp = MLPClassifier(solver = solverType, hidden_layer_sizes=(layers), activation = activationType)
		model = mlp.fit(X_train, Y_train)
		return model


	def predictions(model, X_test, Y_test):
		
		pred = model.predict(X_test)
		return pred
