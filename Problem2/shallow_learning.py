
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

CLASS_PATH = './RESULTS/DATA_CLASSES/DATA_CLASS_SMALL'
MODEL_PATH = './RESULTS/SHALLOW/SAVED_MODELS_SHALLOW/'
REPORT_PATH = './RESULTS/SHALLOW/CLASSIFICATION_REPORT/'
DataClass = data_preprocessing.loadDataHandler(CLASS_PATH)


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
        'C': scipy.stats.expon(scale=100),
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
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


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
        'n_estimators': scipy.stats.randint(100, 500),
        'max_depth': scipy.stats.randint(10, 50),
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
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred



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
        'n_estimators': scipy.stats.randint(100, 500),
        'max_leaf_nodes': scipy.stats.randint(50, 200),
        'min_samples_leaf': scipy.stats.randint(10, 100),
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
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


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
    with open(f"{REPORT_PATH}/{model_name}_{class_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred




def execute_all_models(data_class):
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


execute_all_models(DataClass)

