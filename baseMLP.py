import sklearn.neural_network

def baseMLP(X_train, Y_train, X_test):

    def model(X_train, Y_train):
        MLP = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 1), activation='logistic', solver='sgd')
        model = MLP.fit(X_train,Y_train)
        return model

    def predictions(model):
        predictions = model.predict(X_test)
        return predictions

    model = model(X_train, Y_train)
    pred = predictions(model)
    return pred
