from sklearn import tree

def bestDT(X_train, Y_train, X_test):

	def base_model(X_train, Y_train, X_test, criterion="gini", depth=None, min_samples=2, impurity=0.0, balanced=None):
		d = 10 if depth == 10 else None
		clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=min_samples, min_impurity_decrease=impurity, class_weight=balanced)
		model = clf.fit(X_train, Y_train)
		return model


	def predictions(models, keys):
		
		out = {}
		for model, key in zip(models, keys):
			pred = model.predict(X_test)
			out[key] = pred

		return out

	models = []
	keys = []

	for i in [x / 100.0 for x in range(0, 60, 5)]:
		for min_sample in range(2, 21):
			base = base_model(X_train, Y_train, X_test, criterion="entropy", min_samples=min_sample)
			depth10_ent = base_model(X_train, Y_train, X_test, criterion="entropy", depth=10, min_samples=min_sample)
			bal_base = base_model(X_train, Y_train, X_test, criterion="entropy", balanced="balanced", min_samples=min_sample)
			bal_depth10_ent = base_model(X_train, Y_train, X_test, criterion="entropy", balanced="balanced", depth=10, min_samples=min_sample)
			
			gini = base_model(X_train, Y_train, X_test, min_samples=min_sample)
			depth10_gini = base_model(X_train, Y_train, X_test, depth=10, min_samples=min_sample)
			bal_gini = base_model(X_train, Y_train, X_test, balanced="balanced", min_samples=min_sample)
			bal_depth10_gini = base_model(X_train, Y_train, X_test, balanced="balanced",depth=10, min_samples=min_sample)

			models += [base, depth10_ent, bal_base, bal_depth10_ent, gini, depth10_gini, bal_gini, bal_depth10_gini]
			keys += [f"base-{i}-{min_sample}", f"depth10_ent-{i}-{min_sample}", f"bal_base-{i}-{min_sample}", 
					f"bal_depth10_ent-{i}-{min_sample}", f"gini-{i}-{min_sample}", f"depth10_gini-{i}-{min_sample}", 
					f"bal_gini-{i}-{min_sample}", f"bal_depth10_gini-{i}-{min_sample}"]

	# models = [base, gini, depth10_ent, depth10_gini]

	pred = predictions(models, keys)

	return pred