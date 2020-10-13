from sklearn import tree

def baseDT(X_train, Y_train, X_test):

	def model(X_train, Y_train, X_test):
		clf = tree.DecisionTreeClassifier(criterion="entropy")
		model = clf.fit(X_train, Y_train)
		return model

	def predictions(model):
		pred = model.predict(X_test)
		return pred

	model = model(X_train, Y_train, X_test)
	pred = predictions(model)
	return pred