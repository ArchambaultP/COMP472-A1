from sklearn.naive_bayes import GaussianNB



def gaussianNB(X_train, Y_train, X_test):
	

	def model():
		
		gnb = GaussianNB()
		model = gnb.fit(X_train, Y_train)
		return model

	def predictions(model):
		
		pred = model.predict(X_test)
		return pred


	model = model()
	pred = predictions(model)
	return pred

