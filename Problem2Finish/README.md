# PROBLEM 2

## Table of Contents
1. [Introduction](#introduction)
2. [Preconditions](#preconditions)
3. [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Execution](#execution)
    - [Visualization](#visualization)
    - [Shallowlearning Models](#Shallowlearning)
    - [RNN](#RNN-RANDOMSEARCH)
4. [Files Description](#files-description)
5. [Notes](#Notes)
6. [Installation](#installation)

   
## Introduction
This project is about developing robust models for text classification on Amazon Review Data. The data can be found at "https://nijianmo.github.io/amazon/index.html". To replicate the approach that was taking doing the report, the json files for these categories are required:
    - Magazine_Subscriptions 
    - Clothing_Shoes_and_jewelry 
    - All_Beauty
    - Arts_Crafts_and_Sewing 
    - AMAZON_FASHION
    - Luxury_Beauty (Test data)
 


## Preconditions
- Ensure that the necessary JSON files for the given categories are available as they are required to perform any operations in the project.

## Usage
### Data Preparation
- Execute `create_and_save_data_classes(CATEGORIES)` using `main.py` to generate the data classes used in the project. 

### Execution
- Run the `main.py` file to execute the entire workflow of the project.

### Visualization
- Use `save_all_plot()` to save all the plots used in the project. To visualize the data.
      - Plot: The distribtuion of number of words in each review
      - Plot: The ratio of the reviews
      - Plot: Top 10 common words in the dataset
      - Plot: Average numbers of words for 5 categories
  
### Shalowlearning Models
- Use `execute_all_shallow_models` to train and save all shallow learning models for dataset
      - SVM
      - Boosting
      - Decision Tree
      - Random Forest
  
#### Ensemble Learning
- Use `ensemble_predictions_majority_priortize_SVM(DataClass, list[Of Models])` to execute ensemble prediction for given shallow learning models. SVM need to be given, as it will be used, as a tie breaker. If tie, SVM models prediction will be used.

### RNN
- RNN model, is defined in modelClass.py, as a class. To initialize class, the datahandler, from data_preprocessing.py is required. 

#### Random Search
- use `run_best_wandb_mode(DataClass)` to train the Hyperparameter Search, for the given sweep config (in Main file).

#### Run best RNN model
- use `wandb_save_model():` Initialize the best model found at wandb (online).
- use `wandb_load_model():` To load the model 

## Files Description
- `main.py`: The entry point for the project. It coordinates data preprocessing, model operations, and visualization.
- `modelClass.py`: Contains the definition and functionality of the Machine Learning model.
- `data_preprocessing.py`: Responsible for the initial processing and preparation of data.
- `shallow_learning.py`: Manages the implementation of shallow learning algorithms.
- `visualization.py`: Provides functions for visualizing data and model outputs.

## NOTES
- There is an error in the way datahandler handle the test data. It does not change the labels from 1-5, to 0-4. It therefore require, to do that action, when the test data is used to evalaute.
- The DataClass and DataClassUni, which is used during the report, do not use the same size of json files. Due to, wanting the same size for both number of reviews in same classes. 

## Installation

| Package           | Version   |
|-------------------|-----------|
| absl-py           | 2.0.0     |
| anyascii          | 0.3.2     |
| appdirs           | 1.4.4     |
| astunparse        | 1.6.3     |
| beautifulsoup4    | 4.12.2    |
| bs4               | 0.0.1     |
| cachetools        | 5.3.1     |
| certifi           | 2023.7.22 |
| charset-normalizer| 3.3.0     |
| click             | 8.1.7     |
| contourpy         | 1.2.0     |
| contractions      | 0.1.73    |
| contradict        | 2.0.1     |
| cycler            | 0.12.1    |
| Data-Transformer  | 0.1.1     |
| datatransform     | 1.7       |
| docker-pycreds    | 0.4.0     |
| flatbuffers       | 23.5.26   |
| fonttools         | 4.47.0    |
| frozendict        | 2.4.0     |
| gast              | 0.5.4     |
| gitdb             | 4.0.11    |
| GitPython         | 3.1.40    |
| google-auth       | 2.23.3    |
| google-auth-oauthlib | 1.0.0 |
| google-pasta      | 0.2.0     |
| grpcio            | 1.59.0    |
| h5py              | 3.10.0    |
| hipster           | 3.0.5     |
| html5lib          | 1.1       |
| idna              | 3.4       |
| Jinja2            | 3.1.2     |
| joblib            | 1.3.2     |
| keras             | 2.14.0    |
| kiwisolver        | 1.4.5     |
| libclang          | 16.0.6    |
| lxml              | 4.9.3     |
| Markdown          | 3.5       |
| MarkupSafe        | 2.1.3     |
| matplotlib        | 3.8.2     |
| ml-dtypes         | 0.2.0     |
| multitasking      | 0.0.11    |
| nltk              | 3.8.1     |
| numpy             | 1.26.0    |
| oauthlib          | 3.2.2     |
| opt-einsum        | 3.3.0     |
| packaging         | 23.2      |
| pandas            | 2.1.1     |
| peewee            | 3.17.0    |
| Pillow            | 10.1.0    |
| pip               | 23.2.1    |
| protobuf          | 4.24.4    |
| psutil            | 5.9.7     |
| pyahocorasick     | 2.0.0     |
| pyasn1            | 0.5.0     |
| pyasn1-modules    | 0.3.0     |
| pyparsing         | 3.1.1     |
| python-dateutil   | 2.8.2     |
| pytz              | 2023.3.post1 |
| PyYAML            | 6.0.1     |
| readerwriterlock  | 1.0.9     |
| regex             | 2023.12.25 |
| requests          | 2.31.0    |
| requests-oauthlib | 1.3.1     |
| rsa               | 4.9       |
| scikit-learn      | 1.3.1     |
| scipy             | 1.11.3    |
| seaborn           | 0.13.1    |
| sentry-sdk        | 1.39.1    |
| setproctitle      | 1.3.3     |
| setuptools        | 68.0.0    |
| six               | 1.16.0    |
| smmap             | 5.0.1     |
| soupsieve         | 2.5       |
| tensorboard       | 2.14.1    |
 &#8203;``【oaicite:0】``&#8203;



