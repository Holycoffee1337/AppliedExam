"""
File: shallow_learning.py 
Purpose: provides shallow learning models,
that provide training and evaluation. 
For the models: SVM, DT, RF, BOOSTING
"""

import scipy.stats
import time
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import data_preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterSampler, GridSearchCV
from scipy.stats import randint as sp_randint
import joblib
import numpy as np

MODEL_PATH = './RESULTS/SHALLOW/SAVED_MODELS_SHALLOW/'
REPORT_PATH = './RESULTS/SHALLOW/CLASSIFICATION_REPORT/'
VECTORIZER_PATH = "./RESULTS/SHALLOW/SAVED_SHALLOW_VECTORIZER/"
REPORT_TEST_PATH = "./RESULTS/SHALLOW/CLASSIFICATION_TEST_REPORT/"

# Suppport Vector Machine
def SVM(DataClass):
    model_name = 'SVM'
    class_name = DataClass.get_class_name
    start_time = time.time()

    # Extract data
    df_train = DataClass.get_combined_train_data()
    df_val = DataClass.get_combined_val_data()

    X_train = df_train['text']
    X_val = df_val['text']
    y_train = df_train['overall']
    y_val = df_val['overall']

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Initialize the SVM model
    svm = SVC()

    # Define the parameter distribution for randomized search
    param_distributions = {
        'C': scipy.stats.expon(scale=10),
        'gamma': scipy.stats.expon(scale=0.1),
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(svm, param_distributions=param_distributions, n_iter=3, cv=5, verbose=10, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Predict and evaluate on the validation set using the best estimator
    best_model = random_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    # Save stats
    end_time = time.time()
    training_duration = end_time - start_time
    report = classification_report(y_val, y_val_pred)
    report = f"Model Training Time: {training_duration:.2f} seconds\n\n" + report

    best_params = random_search.best_params_
    report += "\nBest Parameters:\n"
    for param, value in best_params.items():
        report += f"{param}: {value}\n"

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(report)

    # Save the best estimator and report
    dump(best_model, f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    dump(vectorizer, f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


# Random Forest
def RF(DataClass):
    model_name = 'RF'
    class_name = DataClass.get_class_name
    start_time = time.time()

    # Extract data 
    df_train = DataClass.get_combined_train_data()
    df_val = DataClass.get_combined_val_data()

    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Initialize the RF model
    rf = RandomForestClassifier()

    # Define the parameter distribution for randomized search
    param_distributions = {
        'n_estimators': scipy.stats.randint(10, 100),
        'max_depth': scipy.stats.randint(3, 10),
        'min_samples_split': scipy.stats.randint(2, 20)
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=3, cv=5, verbose=10, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Predict and evaluate on the validation set using the best estimator
    best_model = random_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    # Save STATS
    report = classification_report(y_val, y_val_pred)
    end_time = time.time()
    training_duration = end_time - start_time
    report = classification_report(y_val, y_val_pred)
    report = f"Model Training Time: {training_duration:.2f} seconds\n\n" + report


    # Append best parameters to the report
    best_params = random_search.best_params_
    report += "\nBest Parameters:\n"
    for param, value in best_params.items():
        report += f"{param}: {value}\n"

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(report)

    # Save the best estimator and report
    dump(best_model, f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    dump(vectorizer, f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


# Boosting
def Boosting(DataClass):
    model_name = 'BOOSTING'
    class_name = DataClass.get_class_name
    start_time = time.time()

    # Extract data 
    df_train = DataClass.get_combined_train_data()
    df_val = DataClass.get_combined_val_data()

    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Initialize the Boosting model
    gbt = GradientBoostingClassifier()

    # Define the parameter distribution for randomized search
    param_distributions = {
        'n_estimators': scipy.stats.randint(10, 100),
        'max_leaf_nodes': scipy.stats.randint(8, 30),
        'min_samples_leaf': scipy.stats.randint(5, 20),
        'learning_rate': scipy.stats.uniform(0.001, 0.1)
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(gbt, param_distributions, n_iter=3, cv=5, verbose=10, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Predict and evaluate on the validation set using the best estimator
    best_model = random_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    # Save stats
    report = classification_report(y_val, y_val_pred)
    end_time = time.time()
    training_duration = end_time - start_time
    report = classification_report(y_val, y_val_pred)
    report = f"Model Training Time: {training_duration:.2f} seconds\n\n" + report


    # Append best parameters to the report
    best_params = random_search.best_params_
    report += "\nBest Parameters:\n"
    for param, value in best_params.items():
        report += f"{param}: {value}\n"

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(report)

    # Save the best estimator and report
    dump(best_model, f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    dump(vectorizer, f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


# Decisions Tree
def DT(DataClass):
    model_name = 'DT'
    class_name = DataClass.get_class_name
    start_time = time.time()

    # Extract data 
    df_train = DataClass.get_combined_train_data()
    df_val = DataClass.get_combined_val_data()

    X_train = df_train['text']
    X_val = df_val['text'] 
    y_train = df_train['overall']
    y_val = df_val['overall']

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Initialize Decision Tree model
    decision_tree = DecisionTreeClassifier()

    # Define the parameter distribution for randomized search
    param_distributions = {
        'max_depth': scipy.stats.randint(10, 50),
        'min_samples_split': scipy.stats.randint(2, 20),
        'min_samples_leaf': scipy.stats.randint(1, 10)
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(decision_tree, param_distributions, n_iter=3, cv=5, verbose=10, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Predict and evaluate on the validation set using the best estimator
    best_model = random_search.best_estimator_
    y_val_pred = best_model.predict(X_val)


    # SAVE STATS
    report = classification_report(y_val, y_val_pred)
    end_time = time.time()
    training_duration = end_time - start_time
    report = classification_report(y_val, y_val_pred)
    report = f"Model Training Time: {training_duration:.2f} seconds\n\n" + report


    # Append best parameters to the report
    best_params = random_search.best_params_
    report += "\nBest Parameters:\n"
    for param, value in best_params.items():
        report += f"{param}: {value}\n"

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(report)

    # Save the best estimator and report
    dump(best_model, f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    dump(vectorizer, f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred



# Train all shallow models, and save them 
def execute_all_shallow_models(data_class):
    print("Executing SVM Model")
    SVM(data_class)
    print("SVM Model Execution Completed")

    print("Executing RF Model")
    RF(data_class)
    print("RF Model Execution Completed")

    print("Executing Boosting Model")
    Boosting(data_class)
    print("Boosting Model Execution Completed")

    print("Executing DT Model")
    DT(data_class)
    print("DT Model Execution Completed")



def SVM_evaluate(DataClass):
    model_name = 'SVM'
    class_name = DataClass.get_class_name

    model = joblib.load(f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    vectorizer = joblib.load(f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")

    df_test = DataClass.get_test_data()
    
    test_data = df_test['text']
    test_labels = df_test['overall']

    # Text Vectorization
    test_data = vectorizer.transform(test_data)

    y_pred = model.predict(test_data)
    report = classification_report(test_labels, y_pred)

    # Save the report to a file
    with open(f"{REPORT_TEST_PATH}/{model_name}_classification_report_{class_name}.txt", "w") as file:
        file.write(report)

    print(classification_report(test_labels, y_pred))


# RF Evaluate on test data
def RF_evaluate(DataClass):
    model_name = 'RF'
    class_name = DataClass.get_class_name

    model = joblib.load(f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    vectorizer = joblib.load(f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")

    df_test = DataClass.get_test_data()
    
    test_data = df_test['text']
    test_labels = df_test['overall']

    # Text Vectorization
    test_data = vectorizer.transform(test_data)

    y_pred = model.predict(test_data)
    report = classification_report(test_labels, y_pred)

    # Save the report to a file
    with open(f"{REPORT_TEST_PATH}/{model_name}_classification_report_{class_name}.txt", "w") as file:
        file.write(report)

    print(classification_report(test_labels, y_pred))

# Boosting Evaluate on test data
def Boosting_evaluate(DataClass):
    model_name = 'Boosting'
    class_name = DataClass.get_class_name

    model = joblib.load(f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    vectorizer = joblib.load(f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")

    df_test = DataClass.get_test_data()
    
    test_data = df_test['text']
    test_labels = df_test['overall']

    # Text Vectorization
    test_data = vectorizer.transform(test_data)

    y_pred = model.predict(test_data)
    report = classification_report(test_labels, y_pred)

    # Save the report to a file
    with open(f"{REPORT_TEST_PATH}/{model_name}_classification_report_{class_name}.txt", "w") as file:
        file.write(report)

    print(classification_report(test_labels, y_pred))


# DT Evaluate on test data
def DT_evaluate(DataClass):
    model_name = 'DT'
    class_name = DataClass.get_class_name

    model = joblib.load(f"{MODEL_PATH}/{model_name}_{class_name}.joblib")
    vectorizer = joblib.load(f"{VECTORIZER_PATH}/{model_name}_vectorizer_{class_name}.joblib")

    df_test = DataClass.get_test_data()
    
    test_data = df_test['text']
    test_labels = df_test['overall']

    # Text Vectorization
    test_data = vectorizer.transform(test_data)

    y_pred = model.predict(test_data)
    report = classification_report(test_labels, y_pred)

    # Save the report to a file
    with open(f"{REPORT_TEST_PATH}/{model_name}_classification_report_{class_name}.txt", "w") as file:
        file.write(report)

    print(classification_report(test_labels, y_pred))


# Evaluate all shallow models on test data (Just used for curiosity)
def evaluate_all_shallow_models(DataClass):
    print('')
    print('###### Evaluating RF #####')
    RF_evaluate(DataClass)
    print('')
    print('###### Evaluating Boosting #####')
    Boosting_evaluate(DataClass)
    print('')
    print('###### Evaluating SVM #####')
    SVM_evaluate(DataClass)
    print('')
    print('###### Evaluating DT #####')
    DT_evaluate(DataClass)


# Ensemble SVM and RF models
def ensemble_predictions(shallow_model, rnn_model, X_val):

    # Get probability predictions for each model
    shallow_probs = shallow_model.predict_proba(X_val) 
    rnn_probs = rnn_model.predict(X_val)  

    # Combine predictions by averaging
    combined_probs = (shallow_probs + rnn_probs) / 2

    # Convert probabilities to final predictions
    final_predictions = np.argmax(combined_probs, axis=1)

    return final_predictions



