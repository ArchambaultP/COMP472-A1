from sklearn.naive_bayes import GaussianNB



def gaussianNB(X_train, Y_train, X_test, Y_test):
	

	def model(X_train, Y_train):
		gnb = GaussianNB()
		model = gnb.fit(X_train, Y_train)
		return model

	def predictions(model, X_test, Y_test):
		
		pred = model.predict(X_test)
		return pred

	model = model(X_train, Y_train)
	pred = predictions(model, X_test, Y_test)
	return pred

