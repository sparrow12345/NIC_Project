# import packages
import pickle

import numpy as np
from sklearn.datasets import make_regression
# import models
from sklearn.linear_model import Ridge
# import metrics and scoring modules
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
# import tuning modules
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split

score_function = {"mse": mean_squared_error, "mae": mean_absolute_error, "r2": r2_score, 'rmse':
    lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False), "msle": mean_squared_log_error}

name_score_mapper = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
                     "r2": "r2", "msle": "neg_mean_squared_log_error"}

# constant used for cross validation
CV = KFold(n_splits=5, shuffle=True, random_state=11)


def tune_model(model, params_grid, X_train, y_train, cv=None, scoring='neg_mean_absolute_error'):
    # if a user-friendly name is given, map it to the official one used by sklearn
    if scoring in name_score_mapper:
        scoring = name_score_mapper[scoring]

    if cv is None:
        cv = CV

    searcher = GridSearchCV(model, param_grid=params_grid, cv=cv, scoring=scoring, n_jobs=-1)
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_


def evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, train=True, metrics=None):
    # set the default metric
    if metrics is None:
        metrics = ['mse']

    if isinstance(metrics, str):
        metrics = [metrics]

    if 'msle' in metrics and (y_train <= 0).any():
        # msle cannot be used for target variables with non-positive values
        metrics.remove('msle')

    # train the model
    if train:
        tuned_model.fit(X_train, y_train)

    # predict on the test dataset
    y_pred = tuned_model.predict(X_test)
    # evaluate the model
    scores = dict(list(zip(metrics, [score_function[m](y_test, y_pred) for m in metrics])))
    return tuned_model, scores


def save_model(tuned_model, path):
    with open(path, 'wb') as f:
        pickle.dump(tuned_model, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def apply_model(model, X_train, X_test, y_train, y_test, params_grid, save=True, save_path=None, test_size=0.2 ,
                tune_metric=None, test_metrics=None, cv=None):
    # the dataset passed is assumed to be ready to be processed
    # all its features are numerical and all its missing values are imputed/discarded

    if save and save_path is None:
        raise ValueError("Please pass a path to save the model or set the 'save' parameter to False")

    # tune the model
    tuned_model = tune_model(model, params_grid, X_train, y_train, cv=cv, scoring=tune_metric)

    # evaluate teh tuned model
    model, results = evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, metrics=test_metrics)
    # save the model to the passed path
    if save:
        save_model(tuned_model, save_path)

    return model, results


def try_model(model, X, y, params_grid, save=True, save_path=None, test_size=0.2, tune_metric=None,
              test_metrics=None, cv=None):
    # the dataset passed is assumed to be ready to be processed
    # all its features are numerical and all its missing values are imputed/discarded

    if save and save_path is None:
        raise ValueError("Please pass a path to save the model or set the 'save' parameter to False")

    # split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=11)

    # tune the model
    tuned_model = tune_model(model, params_grid, X_train, y_train, cv=cv, scoring=tune_metric)

    # evaluate teh tuned model
    model, results = evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, metrics=test_metrics)
    # save the model to the passed path
    if save:
        save_model(tuned_model, save_path)

    return model, results


ridge_basic = Ridge(max_iter=5000)

ridge_grid = {"alpha": np.logspace(0.001, 10, 20)}


def try_ridge(X, y, lr_model=ridge_basic, params_grid=None, save=True, save_path=None,
              test_size=0.2, tune_metric=None, test_metrics=None, cv=None):
    if params_grid is None:
        params_grid = ridge_grid

    return try_model(lr_model, X, y, params_grid, save=save, save_path=save_path,
                     test_size=test_size, tune_metric=tune_metric, test_metrics=test_metrics, cv=cv)


if __name__ == "__main__":
    X, Y = make_regression(n_samples=4000, n_features=20, random_state=18, n_informative=8)

    lr, results = try_ridge(X, Y, save=False, test_metrics=['mse', 'rmse', 'r2', "msle"], tune_metric='mse')

    print(results)
