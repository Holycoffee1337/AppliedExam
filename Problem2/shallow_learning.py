# Shallow Learning Modeller
# Models we have learning about
# Decisions Tree
# Ensembling
# Boosting 
# Support Vector Machiens

# Support Vector Machines
# Implement Grid Search
import data_preprocessing
from shallowmodels import svm
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble 
import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from sklearn import tree


# DataHandler = data_preprocessing.DataHandler
DataClass = data_preprocessing.loadDataHandler('Data_Class_uni')
DataHandler: data_preprocessing.DataHandler = DataClass
DataHandler.remove_stop_words_in_combined()



#TODO: Do so the models take the test data??
#TODO: Maybe use pipeline on all models, for efficiency? 


def SVMPipeline(DataClass: DataHandler):
    # Extract training data
    X_train = DataClass.get_combined_train_data()
    y_train = X_train['overall']
    X_train_texts = X_train['text']

    # Define a pipeline combining a text vectorizer with an SVM classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC())
    ])

    # Define the parameter grid to search
    param_grid = {
        'tfidf__max_features': [500, 1000],  # Example range for max_features
        'svm__C': [0.1, 1, 10],              # Example range for C
        'svm__kernel': ['linear', 'rbf']     # Example kernels
    }

    # Initialize the grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train_texts, y_train)

    # Best estimator found by grid search
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validated score: {grid_search.best_score_}")

    # Save the best estimator
    folder_path = "./shallowmodels/"
    best_model = grid_search.best_estimator_
    dump(best_model, f"{folder_path}/svm_pipeline_model.joblib")

    # Extract validation data
    X_val = DataClass.get_combined_val_data()
    y_val = X_val['overall']
    X_val_texts = X_val['text']

    # Predict and evaluate on the validation set using the best estimator
    y_val_pred = best_model.predict(X_val_texts)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    return y_val_pred



def SVM(DataClass: DataHandler):

    # Extract data 
    df_train = DataHandler.get_combined_train_data()
    df_val = DataHandler.get_combined_val_data()
    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']


    # Add Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)


    # Initialize the grid search
    svm = SVC()
    param_grid = {
            'C': [0.1, 1, 10],          # C: Regularization parameter (more freedom to sample)
        'kernel': ['linear', 'rbf']    # Kernels 
    }
    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)


    # Predict and evaluate on the validation set using the best estimator
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))


    # Save the best estimator
    print(grid_search.best_params_)
    # folder_path = "./shallowmodels/"
    # dump(best_model, f"{folder_path}/svm_pipeline_model.joblib")

    return y_val_pred




def RF(DataClass: DataHandler):

    # Extract data 
    df_train = DataHandler.get_combined_train_data()
    df_val = DataHandler.get_combined_val_data()
    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']


    # Add Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)


    # Gridsearch on model    
    rf = ensemble.RandomForestClassifier()
    param_grid = [{'n_estimators': [100, 200, 300], 
                   'max_depth': [10, 20, 30], 
                   'min_samples_split': [2, 10, 20]}]

    grid_search = GridSearchCV(rf, param_grid, refit = True, verbose = 3, n_jobs=-1)


    # Fit
    grid_search.fit(X_train, y_train)
    grid_predictions = grid_search.predict(X_val) 
    print(grid_search.best_params_)

    # Predict and evaluate on the validation set using the best estimator
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))


    # Saving model
    # folder_path = "./results"
    # dump(rf, f"{folder_path}/{model_name}.joblib")
    return grid_predictions 



def Boosting(DataClass: DataHandler):

    # Extract data 
    df_train = DataHandler.get_combined_train_data()
    df_val = DataHandler.get_combined_val_data()
    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']


    # Add Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)


    # Gridsearch on model
    gbt = ensemble.GradientBoostingClassifier(n_estimators = 200)
    param_grid = [{'n_estimators': [100, 200, 300], 
                   'max_leaf_nodes': [80, 100, 150], 
                   'min_samples_leaf': [20, 50, 80], 
                   'learning_rate': [0.1, 0.01, 0.001]}]
    grid_search = GridSearchCV(gbt, param_grid, refit = True, verbose = 3,n_jobs=-1)


    # Fit
    grid_search.fit(X_train, y_train)
    grid_predictions = grid_search.predict(X_val) 
    print(grid_search.best_params_)


    # Predict and evaluate on the validation set using the best estimator
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))


    # Saving model
    print(grid_search.best_params_)
    # folder_path = "./results"
    # dump(gbt, f"{folder_path}/{model_name}.joblib")
    return grid_predictions 



def DT(DataClass: DataHandler):

    # Extract data 
    df_train = DataHandler.get_combined_train_data()
    df_val = DataHandler.get_combined_val_data()
    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']


    # Add Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)


    # Gridsearch on model
    decision_tree = tree.DecisionTreeClassifier()
    param_grid = [{'n_estimators': [100, 200, 300], 
                   'max_leaf_nodes': [80, 100, 150], 
                   'min_samples_leaf': [20, 50, 80], 
                   'learning_rate': [0.1, 0.01, 0.001]}]
    grid_search = GridSearchCV(decision_tree, param_grid, refit = True, verbose = 3,n_jobs=-1)


    # Fit
    grid_search.fit(X_train, y_train)
    grid_predictions = grid_search.predict(X_val) 
    print(grid_search.best_params_)


    # Predict and evaluate on the validation set using the best estimator
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))


    # Saving model
    print(grid_search.best_params_)
    # folder_path = "./results"
    # dump(gbt, f"{folder_path}/{model_name}.joblib")
    return grid_predictions 



    

SVM(DataClass)
