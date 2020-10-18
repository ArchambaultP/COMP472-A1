from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score,plot_confusion_matrix
import matplotlib.pyplot as plt

def baseDT(X_train, Y_train, X_test, Y_test):

    def model(X_train, Y_train, X_test):
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        model = clf.fit(X_train, Y_train)
        return model

    def predictions(model):
        pred = model.predict(X_test)
        return pred

    model = model(X_train, Y_train, X_test)
    pred = predictions(model)
    print(f"Accuracy: {accuracy_score(Y_test, pred)}")
    print(f"Micro: {f1_score(Y_test, pred, average='micro')}")
    print(f"Macro: {f1_score(Y_test, pred, average='macro')}")
    plot_confusion_matrix(model, X_test, Y_test)
    plt.show()

    return pred