from sklearn.naive_bayes import GaussianNB



def gaussianNB(X_train, Y_train, X_test):
	
	#creating the model
	def model():
		
		gnb = GaussianNB()
		model = gnb.fit(X_train, Y_train)
		return model

	#predicting values
	def predictions(model):
		
		pred = model.predict(X_test)
		return pred


	model = model()
	pred = predictions(model)
	return pred

