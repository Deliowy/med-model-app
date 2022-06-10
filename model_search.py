import pickle
import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_FILENAME = "База_9"


def stream_to_file(output):
    with open(f"{BASE_FILENAME}_model_search.txt", "a") as writer:
        for parameter in output:
            if type(parameter) is dict:
                pprint.pprint(parameter, writer)
            else:
                writer.write(str(parameter))
                writer.write("\n")


def get_metrics(X, Y, estimator):
    metrics_df = {
        "Train_Accuracy": [],
        "Train_Precision": [],
        "Train_Recall": [],
        "Train_Specificity": [],
        "Train_F1": [],
        "Train_ROC-AUC": [],
        "Train_PRC-AUC": [],
        "Test_Accuracy": [],
        "Test_Precision": [],
        "Test_Recall": [],
        "Test_Specificity": [],
        "Test_F1": [],
        "Test_ROC-AUC": [],
        "Test_PRC-AUC": [],
    }

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=40
    )

    Y_pred = estimator.predict(X_train)
    accuracy = accuracy_score(Y_train, Y_pred)
    metrics_df["Train_Accuracy"].append(accuracy)

    precision = precision_score(Y_train, Y_pred)
    metrics_df["Train_Precision"].append(precision)

    recall = recall_score(Y_train, Y_pred)
    metrics_df["Train_Recall"].append(recall)

    specificity = specificity_score(Y_train, Y_pred)
    metrics_df["Train_Specificity"].append(specificity)

    f1 = f1_score(Y_train, Y_pred)
    metrics_df["Train_F1"].append(f1)

    Y_score = estimator.predict_proba(X_train)[:, 1]
    roc_auc = roc_auc_score(Y_train, Y_score)
    metrics_df["Train_ROC-AUC"].append(roc_auc)

    prc_auc = average_precision_score(Y_train, Y_score)
    metrics_df["Train_PRC-AUC"].append(prc_auc)

    Y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    metrics_df["Test_Accuracy"].append(accuracy)

    precision = precision_score(Y_test, Y_pred)
    metrics_df["Test_Precision"].append(precision)

    recall = recall_score(Y_test, Y_pred)
    metrics_df["Test_Recall"].append(recall)

    specificity = specificity_score(Y_test, Y_pred)
    metrics_df["Test_Specificity"].append(specificity)

    f1 = f1_score(Y_test, Y_pred)
    metrics_df["Test_F1"].append(f1)

    Y_score = estimator.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(Y_test, Y_score)
    metrics_df["Test_ROC-AUC"].append(roc_auc)

    prc_auc = average_precision_score(Y_test, Y_score)
    metrics_df["Test_PRC-AUC"].append(prc_auc)

    metrics_df = pd.DataFrame(metrics_df)

    return metrics_df

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def prc_auc_score(estimator, X, y):
    y_score = estimator.predict_proba(X)[:, 1]
    prc_auc = average_precision_score(y, y_score)
    return prc_auc

def DT_model_search(df: pd.DataFrame):

    print("DecisionTree")

    output_to_file = []
    output_to_file.append("DecisionTree")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
    
    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score
    }

    parametrs = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": np.arange(2, 50, 2),
        "max_features": np.arange(3, 26, 2),
        "min_samples_split": np.arange(2, 100, 5),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    output_to_file.append("GridSearchCV")

    clf = DecisionTreeClassifier(random_state=40)
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "DTree_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = DecisionTreeClassifier()
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "DTree_RandomSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


def RF_model_search(df: pd.DataFrame):

    output_to_file = []
    output_to_file.append("\n")
    output_to_file.append("RandomForest")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, Y_train = X_train[mask, :], Y_train[mask]

    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score,
    }

    parametrs = {
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(3, 26, 2),
        "n_estimators": np.arange(10, 100, 10),
        "max_depth": np.arange(2, 50, 3),
        "min_samples_split": np.arange(2, 100, 5),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    clf = RandomForestClassifier(n_jobs=-1)
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "RForest_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = RandomForestClassifier(n_jobs=-1)
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "RForest_RandomSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


def KNN_model_search(df: pd.DataFrame):

    output_to_file = []
    output_to_file.append("\n")
    output_to_file.append("KNN")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, Y_train = X_train[mask, :], Y_train[mask]


    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score,
    }

    parametrs = {
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "n_neighbors": np.arange(5, 20, 2),
        "leaf_size": np.arange(10, 61, 10),
        "p": np.arange(1, 8, 2),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    clf = KNeighborsClassifier(n_jobs=-1)
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "KNN_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = KNeighborsClassifier(n_jobs=-1)
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "KNN_RandomSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


def SVM_model_search(df: pd.DataFrame):
    output_to_file = []
    output_to_file.append("\n")
    output_to_file.append("SVM")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, Y_train = X_train[mask, :], Y_train[mask]


    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score,
    }

    parametrs = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["auto", "scale"],
        "class_weight": ["balanced", None],
        'degree': np.arange(1, 6, 1),
        "C": np.arange(0.1, 5.0, 0.1),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    clf = SVC(probability=True)
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "SVM_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = SVC(probability=True)
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "SVM_RandSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


def GradBoost_model_search(df: pd.DataFrame):

    output_to_file = []
    output_to_file.append("\n")
    output_to_file.append("GradBoosting")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, Y_train = X_train[mask, :], Y_train[mask]


    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score,
    }

    parametrs = {
        "loss": ["log_loss", 'exponential'],
        "n_estimators": np.arange(10, 100, 5),
        "learning_rate": np.arange(0.1, 2.5, 0.1),
        "max_features": np.arange(3, 26, 2),
        "min_samples_split": np.arange(2, 100, 5),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    output_to_file.append("GridSearchCV")

    clf = GradientBoostingClassifier()
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "GradBoost_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = GradientBoostingClassifier()
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "GradBoost_RandomSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


def LogisticRegression_model_search(df: pd.DataFrame):

    output_to_file = []
    output_to_file.append("\n")
    output_to_file.append("LogisticRegression")

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, Y_train = X_train[mask, :], Y_train[mask]


    scoring = {
        "AUC": "roc_auc",
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score),
        "Recall": make_scorer(recall_score),
        "F1": make_scorer(f1_score),
        "Specificity": make_scorer(specificity_score),
        "PRC-AUC": prc_auc_score,
    }

    parametrs = {
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "class_weight": ["balanced", None],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "C": np.arange(0.01, 2.5, 0.1),
        "max_iter": np.arange(5, 501, 50),
    }

    output_to_file.append(parametrs)
    output_to_file.append("\n")

    ## GRID SEARCH
    print("GridSearchCV")

    output_to_file.append("GridSearchCV")

    clf = LogisticRegression(n_jobs=-1)
    grid = GridSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
    )

    grid.fit(X_train, Y_train)

    output_to_file.append({"Best Parameters: ": grid.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, grid.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "LogReg_GridSearch.sav"
    pickle.dump(grid.best_estimator_, open(pickle_filename, "wb"))

    ## RANDOMIZED SEARCH
    print("RandomizedSearchCV")

    output_to_file.append("\n")
    output_to_file.append("RandomizedSearchCV")

    clf = LogisticRegression(n_jobs=-1)
    rand = RandomizedSearchCV(
        clf,
        parametrs,
        cv=5,
        scoring=scoring,
        refit="PRC-AUC",
        n_jobs=-1,
        random_state=40,
    )

    rand.fit(X_train, Y_train)
    print(rand.best_params_)

    output_to_file.append({"Best Parameters: ": rand.best_params_})
    output_to_file.append("\n")

    metrics = get_metrics(X, Y, rand.best_estimator_).to_dict()
    output_to_file.append(metrics)

    pickle_filename = "LogReg_RandomSearch.sav"
    pickle.dump(rand.best_estimator_, open(pickle_filename, "wb"))

    stream_to_file(output_to_file)


if __name__ == "__main__":
    print("БАЗА 9 -- Предсказание рецидива заболевания")

    filename = f"{BASE_FILENAME}.xlsx"

    df = pd.read_excel(filename)
    df = df.dropna()

    print(df['Target'].value_counts())

    DT_model_search(df)

    RF_model_search(df)

    KNN_model_search(df)

    SVM_model_search(df)

    GradBoost_model_search(df)

    LogisticRegression_model_search(df)
