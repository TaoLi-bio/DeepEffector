import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, matthews_corrcoef
from collections import Counter


def custom_specificity_score(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp) if (tn + fp) != 0 else 0


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score, zero_division=1),
    'specificity': make_scorer(custom_specificity_score),
    'f1': make_scorer(f1_score, zero_division=1),
    'mcc': make_scorer(matthews_corrcoef)
}


positive_train = pd.read_excel('positive_train')
negative_train = pd.read_excel('negative_train')


positive_train['label'] = 1
negative_train['label'] = 0


X_train = pd.concat([positive_train, negative_train], axis=0).drop(columns=['label'])
y_train = pd.concat([positive_train, negative_train], axis=0)['label']


print(f"Original dataset shape: {Counter(y_train)}")


param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


base_classifier = SVC()


grid_search_svc = GridSearchCV(estimator=base_classifier, param_grid=param_grid_svc,
                               scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)


grid_search_svc.fit(X_train, y_train)


print("Best parameters for SVC:")
print(grid_search_svc.best_params_)
print("Best Accuracy Score for SVC:")
print(grid_search_svc.best_score_)


best_svc = grid_search_svc.best_estimator_


cv_results = cross_validate(best_svc, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)


print(f"Accuracy: {np.mean(cv_results['test_accuracy'])} ± {np.std(cv_results['test_accuracy'])}")
print(f"Sensitivity: {np.mean(cv_results['test_sensitivity'])} ± {np.std(cv_results['test_sensitivity'])}")
print(f"Specificity: {np.mean(cv_results['test_specificity'])} ± {np.std(cv_results['test_specificity'])}")
print(f"F1 Score: {np.mean(cv_results['test_f1'])} ± {np.std(cv_results['test_f1'])}")
print(f"MCC: {np.mean(cv_results['test_mcc'])} ± {np.std(cv_results['test_mcc'])}")


positive_test = pd.read_excel('positive_test')
negative_test = pd.read_excel('negative_test')


positive_test['label'] = 1
negative_test['label'] = 0


X_test = pd.concat([positive_test, negative_test], axis=0).drop(columns=['label'])
y_test = pd.concat([positive_test, negative_test], axis=0)['label']


y_pred = best_svc.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, zero_division=1)
specificity = custom_specificity_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=1)
mcc = matthews_corrcoef(y_test, y_pred)


print("Independent Test Set Results:")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")
