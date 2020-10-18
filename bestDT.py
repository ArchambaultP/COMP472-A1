from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

def bestDT(X_train, Y_train, X_test, Y_test, labels):

    def base_model(X_train, Y_train, X_test, criterion="gini", depth=None, min_samples=2, impurity=0.0, balanced=None):
        d = 10 if depth == 10 else None
        clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=min_samples, min_impurity_decrease=impurity, class_weight=balanced)
        model = clf.fit(X_train, Y_train)
        return model

    def predictions(models, keys):
        out = {}
        for model, key in zip(models, keys):
            pred = model.predict(X_test)
            out[key] = (model, pred)
        return out

    def stats(Y_true, pred):
        f1Micro = f1_score(Y_true, pred, average='micro')
        f1Macro = f1_score(Y_true, pred, average='macro')
        accuracy = accuracy_score(Y_true, pred)
        return (f1Micro + f1Macro + accuracy) / 3

    def rank_models(Y_true, preds, labels):
        rankings = {}
        f1 = {}
        recall = {}
        precision = {}

        for key in preds.keys():
            model = preds[key][0]
            pred = preds[key][1]
            rankings[key] = stats(Y_true, pred)
            p, r, f1_score, _ = precision_recall_fscore_support(Y_true, pred, average="micro")
            precision[key] = p
            recall[key] = r
            f1[key] = f1_score

        rankings = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1], reverse=True)}
        for el in list(rankings.items())[:5]:
            print("Ranked models:")
            print(el)


        return rankings, f1, precision, recall

    tuned_params=[{'criterion':['entropy', 'gini'], 'max_depth':[None, 10], 'class_weight':[None, 'balanced'],     
                    'min_samples_split': [2,3,4,5,6], 'min_impurity_decrease':[0, 0.1]}]
    # tuned_params=[{'criterion':['entropy', 'gini'], 'max_depth':[None, 10], 'class_weight':[None, 'balanced'], 
    #             'min_samples_split': [2,3, 5,6], 'min_impurity_decrease':[0]}]

    # best_dt = GridSearchCV(
    # tree.DecisionTreeClassifier(), tuned_params)
    # best_dt.fit(X_train, Y_train)
    # means = best_dt.cv_results_['mean_test_score']
    # stds = best_dt.cv_results_['std_test_score']
    # for mean, std, params in sorted(zip(means, stds, best_dt.cv_results_['params']), key = lambda x: x[0]):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    # print(best_dt.best_params_)

    n = 20
    acc1 = 0
    macro1 = 0
    micro1 = 0
    acc2 = 0
    macro2 = 0
    micro2 = 0
    preds = [0]

    # for i in range(n):
    #     # best_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_impurity_decrease=0, class_weight='balanced')
    #     # best_dt.fit(X_train, Y_train)
    #     # pred = best_dt.predict(X_test)
    #     # acc1 += accuracy_score(Y_test, pred)
    #     # micro1 += f1_score(Y_test, pred, average='micro')
    #     # macro1 += f1_score(Y_test, pred, average='macro')
    #     # print(f"Accuracy: {accuracy_score(Y_test, pred)}\nMicro f1: {f1_score(Y_test, pred, average='micro')}\nMacro f1: {f1_score(Y_test, pred, average='macro')}")
    #     # print()
    #     best_dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0, class_weight='balanced')
    #     best_dt.fit(X_train, Y_train)
    #     pred = best_dt.predict(X_test)
    #     acc2 += accuracy_score(Y_test, pred)
    #     micro2 += f1_score(Y_test, pred, average='micro')
    #     macro2 += f1_score(Y_test, pred, average='macro')
    #     print(f"Accuracy: {accuracy_score(Y_test, pred)}\nMicro f1: {f1_score(Y_test, pred, average='micro')}\nMacro f1: {f1_score(Y_test, pred, average='macro')}")

    # print(f"Accuracy: {acc1 / n}\nMicro f1: {micro1 / n}\nMacro f1: {macro1 / n}")
    # print()
    # print(f"Accuracy: {acc2 / n}\nMicro f1: {micro2 / n}\nMacro f1: {macro2 / n}")

    # best_dt_dataset1 = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0, class_weight=None)
    # best_dt_dataset1.fit(X_train, Y_train)
    # plot_confusion_matrix(best_dt_dataset1, X_test, Y_test)
    # plt.show()
    # preds = best_dt_dataset1.predict(X_test)

    best_dt_dataset2 = tree.DecisionTreeClassifier(criterion="entropy", class_weight='balanced', min_samples_split=2, min_impurity_decrease=0)
    best_dt_dataset2.fit(X_train, Y_train)
    pred = best_dt_dataset2.predict(X_test)
    print(f"Accuracy: {accuracy_score(Y_test, pred)}")
    print(f"Micro: {f1_score(Y_test, pred, average='micro')}")
    print(f"Macro: {f1_score(Y_test, pred, average='macro')}")
    plot_confusion_matrix(best_dt_dataset2, X_test, Y_test)
    plt.show()


    return pred