from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

def bestDT(X_train, Y_train, X_test, Y_test, labels):


    # tuned_params=[{'criterion':['entropy', 'gini'], 'max_depth':[None, 10], 'class_weight':[None, 'balanced'],     
    #                 'min_samples_split': [2,3,4,5,6], 'min_impurity_decrease':[0, 0.05, 0.1]}]

    # best_dt = GridSearchCV(
    # tree.DecisionTreeClassifier(), tuned_params)
    # best_dt.fit(X_train, Y_train)
    # means = best_dt.cv_results_['mean_test_score']
    # stds = best_dt.cv_results_['std_test_score']
    # for mean, std, params in sorted(zip(means, stds, best_dt.cv_results_['params']), key = lambda x: x[0]):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    # print(best_dt.best_params_)

    best_dt_dataset1 = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0, class_weight=None)
    best_dt_dataset1.fit(X_train, Y_train)
    # plot_confusion_matrix(best_dt_dataset1, X_test, Y_test)
    # plt.show()
    preds = best_dt_dataset1.predict(X_test)

    # best_dt_dataset2 = tree.DecisionTreeClassifier(criterion="entropy", class_weight='balanced', min_samples_split=2, min_impurity_decrease=0)
    # best_dt_dataset2.fit(X_train, Y_train)
    # preds = best_dt_dataset2.predict(X_test)
    # print(f"Accuracy: {accuracy_score(Y_test, preds)}")
    # print(f"Micro: {f1_score(Y_test, preds, average='micro')}")
    # print(f"Macro: {f1_score(Y_test, preds, average='macro')}")
    # plot_confusion_matrix(best_dt_dataset2, X_test, Y_test)
    # plt.show()


    return preds