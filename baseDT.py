from sklearn import tree

def train(X,Y):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)
