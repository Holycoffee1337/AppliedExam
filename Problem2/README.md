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
