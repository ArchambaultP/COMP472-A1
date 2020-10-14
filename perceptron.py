import sklearn.linear_model.Perceptron

def perceptron(X_train, Y_train, X_test):

    def model(X_train, Y_train):
        model = sklearn.linear_model.Perceptron().fit(X_train, Y_train)
        return model

    def predictions(model):
        predictions = model.predict(X_test)
        return predictions

    model = model(X_train, Y_train)
    pred = predictions(model)
    return pred


