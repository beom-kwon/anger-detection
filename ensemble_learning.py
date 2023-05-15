from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa (no quality assurance)
from sklearn.model_selection import HalvingGridSearchCV
from time import time
import numpy as np
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)


def hyperparameter_optimization(name, clf, train_x, train_y):
    pipeline = make_pipeline(StandardScaler(), clf)
    alg_name = pipeline.steps[1][0]
    if name == 'LR':  # Logistic Regression
        param_grid = {'{}__C'.format(alg_name): [0.001, 0.01, 0.1, 1, 10],
                      '{}__class_weight'.format(alg_name): ['balanced', None],
                      '{}__dual'.format(alg_name): [False],
                      '{}__fit_intercept'.format(alg_name): [True, False],
                      '{}__multi_class'.format(alg_name): ['auto', 'ovr'],
                      '{}__penalty'.format(alg_name): ['l1', 'l2'],
                      '{}__solver'.format(alg_name): ['liblinear'],
                      '{}__warm_start'.format(alg_name): [True, False]}
    elif name == 'KNN':  # k-Nearest Neighbors
        param_grid = {'{}__n_neighbors'.format(alg_name): np.arange(3, 12, 2),
                      '{}__weights'.format(alg_name): ['uniform', 'distance'],
                      '{}__algorithm'.format(alg_name): ['auto'],  # 'ball_tree', 'kd_tree', 'brute', 'auto'
                      '{}__leaf_size'.format(alg_name): [30],
                      '{}__p'.format(alg_name): [1, 2, 3],
                      '{}__metric'.format(alg_name): ['minkowski']}
    elif name == 'NB':  # Naive Bayes
        param_grid = {'{}__priors'.format(alg_name): [None],
                      '{}__var_smoothing'.format(alg_name): [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]}
    elif name == 'C-SVM':  # C-Support Vector Classification
        param_grid = {'{}__C'.format(alg_name): [1e-2, 1e-1, 1e0, 1e1, 1e2],
                      '{}__kernel'.format(alg_name): ['poly', 'rbf', 'sigmoid'],
                      '{}__degree'.format(alg_name): [2, 3, 4, 5],
                      '{}__gamma'.format(alg_name): ['scale', 'auto'],
                      '{}__coef0'.format(alg_name): [0.0, 0.3, 0.5],
                      '{}__shrinking'.format(alg_name): [True, False],
                      '{}__decision_function_shape'.format(alg_name): ['ovo', 'ovr']}
    elif name == 'DT':  # Decision Tree
        param_grid = {'{}__criterion'.format(alg_name): ['gini', 'entropy', 'log_loss'],
                      '{}__splitter'.format(alg_name): ['best', 'random'],
                      '{}__max_depth'.format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}

    hgs = HalvingGridSearchCV(pipeline,
                              param_grid=param_grid,
                              factor=3,    # the proportion of candidates selected for each subsequent iteration
                              cv=5,        # 5-fold Cross Validation
                              random_state=42,
                              refit=True)  # If True, refit an estimator using the best parameters

    tic = time()
    hgs.fit(train_x, train_y)
    hgs_time = time() - tic

    accuracy = hgs.score(train_x, train_y)
    # print('({})'.format(name) + ' Acc.: {:.3f}, '.format(accuracy) + 'Time: {:.2f}s'.format(hgs_time))
    return hgs


names = [
    'LR',     # Logistic Regression
    'KNN',    # k-Nearest Neighbors
    'NB',     # Naive Bayes
    'C-SVM',  # C-Support Vector Classification
    'DT']     # Decision Tree

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(),
    DecisionTreeClassifier()]

type_c = 3  # Types of Classifiers (for LR: 0, for KNN: 1, for NB: 2, for C-SVM: 3, for DT: 4)
num_c = 11  # Total Number of Classifiers (for the conventional method : 1, for the proposed method: 3, 5, 7, 9, 11)

input_file_name = 'feature' + str(num_c) + '.npz'
npz_file = np.load(input_file_name)
x = npz_file['x']
y = npz_file['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

print('Training Data: ', x_train.shape, y_train.shape)
print('Test Data: ', x_test.shape, y_test.shape)

print('Classifier Type: ', names[type_c])
print('Number of Classifiers: ', num_c)

if num_c == 1:  # conventional feature extraction technique
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    result_arr = np.reshape(r1, (-1, 1))
elif num_c == 3:
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    model2 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 15:30], y_train)
    model3 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 30:45], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    r2 = model2.predict(x_test[:, 15:30])
    r3 = model3.predict(x_test[:, 30:45])
    result_arr = np.column_stack((r1, r2, r3))
elif num_c == 5:
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    model2 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 15:30], y_train)
    model3 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 30:45], y_train)
    model4 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 45:60], y_train)
    model5 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 60:75], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    r2 = model2.predict(x_test[:, 15:30])
    r3 = model3.predict(x_test[:, 30:45])
    r4 = model4.predict(x_test[:, 45:60])
    r5 = model5.predict(x_test[:, 60:75])
    result_arr = np.column_stack((r1, r2, r3, r4, r5))
elif num_c == 7:
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    model2 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 15:30], y_train)
    model3 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 30:45], y_train)
    model4 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 45:60], y_train)
    model5 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 60:75], y_train)
    model6 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 75:90], y_train)
    model7 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 90:105], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    r2 = model2.predict(x_test[:, 15:30])
    r3 = model3.predict(x_test[:, 30:45])
    r4 = model4.predict(x_test[:, 45:60])
    r5 = model5.predict(x_test[:, 60:75])
    r6 = model6.predict(x_test[:, 75:90])
    r7 = model7.predict(x_test[:, 90:105])
    result_arr = np.column_stack((r1, r2, r3, r4, r5, r6, r7))
elif num_c == 9:
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    model2 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 15:30], y_train)
    model3 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 30:45], y_train)
    model4 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 45:60], y_train)
    model5 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 60:75], y_train)
    model6 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 75:90], y_train)
    model7 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 90:105], y_train)
    model8 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 105:120], y_train)
    model9 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 120:135], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    r2 = model2.predict(x_test[:, 15:30])
    r3 = model3.predict(x_test[:, 30:45])
    r4 = model4.predict(x_test[:, 45:60])
    r5 = model5.predict(x_test[:, 60:75])
    r6 = model6.predict(x_test[:, 75:90])
    r7 = model7.predict(x_test[:, 90:105])
    r8 = model7.predict(x_test[:, 105:120])
    r9 = model7.predict(x_test[:, 120:135])
    result_arr = np.column_stack((r1, r2, r3, r4, r5, r6, r7, r8, r9))
elif num_c == 11:
    model1 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 0:15], y_train)
    model2 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 15:30], y_train)
    model3 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 30:45], y_train)
    model4 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 45:60], y_train)
    model5 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 60:75], y_train)
    model6 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 75:90], y_train)
    model7 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 90:105], y_train)
    model8 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 105:120], y_train)
    model9 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 120:135], y_train)
    model10 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 135:150], y_train)
    model11 = hyperparameter_optimization(names[type_c], classifiers[type_c], x_train[:, 150:165], y_train)
    r1 = model1.predict(x_test[:, 0:15])
    r2 = model2.predict(x_test[:, 15:30])
    r3 = model3.predict(x_test[:, 30:45])
    r4 = model4.predict(x_test[:, 45:60])
    r5 = model5.predict(x_test[:, 60:75])
    r6 = model6.predict(x_test[:, 75:90])
    r7 = model7.predict(x_test[:, 90:105])
    r8 = model7.predict(x_test[:, 105:120])
    r9 = model7.predict(x_test[:, 120:135])
    r10 = model7.predict(x_test[:, 135:150])
    r11 = model7.predict(x_test[:, 150:165])
    result_arr = np.column_stack((r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11))


final_result = []
cnt_tp, cnt_fn, cnt_fp, cnt_tn = 0, 0, 0, 0
for idx in range(result_arr.shape[0]):
    unique, counts = np.unique(result_arr[idx, :], return_counts=True)
    final_result.append(unique[np.argmax(counts)])

    if (y_test[idx] == 1) & (final_result[idx] == 1):
        cnt_tp = cnt_tp + 1
    elif (y_test[idx] == 1) & (final_result[idx] == 0):
        cnt_fn = cnt_fn + 1
    elif (y_test[idx] == 0) & (final_result[idx] == 1):
        cnt_fp = cnt_fp + 1
    elif (y_test[idx] == 0) & (final_result[idx] == 0):
        cnt_tn = cnt_tn + 1

TPR = cnt_tp / (cnt_tp + cnt_fn)
PPV = cnt_tp / (cnt_tp + cnt_fp)
ACC = (cnt_tp + cnt_tn) / (cnt_tp + cnt_tn + cnt_fp + cnt_fn)

print('TPR: %.3f' % TPR)
print('PPV: %.3f' % PPV)
print('F1: %.3f' % (2 * PPV * TPR / (PPV + TPR)))
print('ACC: %.3f' % ACC)
