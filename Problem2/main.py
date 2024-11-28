"""
File: main.py, is used as entry point to the project.

Modules:
    - data_preprocessing.py: Handles the data preparing for the training of the model 
    - modelClass.py: Define the RNN model 
    - shallow_learning.py: Define the shallow learning models, SVM, DT, RF, BOOSTING 
    - visualization.py: Provides functions to visualize the results
"""
import tensorflow as tf
from absl.testing.parameterized import parameters
from keras.src.engine.data_adapter import DataHandler
from tensorflow._api.v2 import train
from tensorflow._api.v2.random import create_rng_state
from wandb.sdk.wandb_sweep import sweep
import modelClass
import wandb
from wandb.keras import WandbCallback
import os
import nltk
from nltk.corpus import stopwords
import time
import keras.models
from keras.models import load_model
import matplotlib as plt
import warnings
from tensorflow.keras.callbacks import ModelCheckpoint
warnings.filterwarnings('ignore')

# Our Files
import data_preprocessing
from data_preprocessing import loadDataHandler
import visualization
from shallow_learning import SVM, DT, Boosting, RF, Boosting_evaluate, SVM_evaluate, ensemble_predictions_majority_prioritize_SVM, evaluate_all_shallow_models
from shallow_learning import ensemble_predictions, RF_evaluate, ensemble_predictions_majority, ensemble_predictions_majority_prioritize_SVM_EVALUATE

##### Initialize wandb #####
os.environ["WANDB_API_KEY"] = ''
project_name = 'HELLO2' 
entity = 'marcs20'             
wandb.login()
##### Set wandb to offline mode #####
# os.environ['WANDB_MODE'] = 'dryrun'
############################


# Sweep config file, for RNN hyperparameter search
sweep_config = {
        'method': 'random',  
        'metric': {
           'name': 'val_accuracy',
           'goal': 'maximize'   
        },
        'parameters': {
            
            'n_epochs': {
                'values': [10] # Using random count anyway
            },

            'lr': {
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            'output_sequence_length':{
                'values': [50, 100]
            },
            'embedding_dimension':{
                'values': [200, 400]
            },

            'lr_scheduler': {
                'values': ['constant', 'decay']
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            

        }
}


# Categories we use for the dataset 
CATEGORIES = [
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', 
              'All_Beauty',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
 

def create_and_save_data_classes(CATEGORIES):
    '''
    This function create the data classes,
    and save them locally in './RESULTS/DATA_CLASSES/'.

        - DATA_CLASS_WITHOUT_STOP_WORDS
        - DATA_CLASS
        - DATA_CLASS_UNI
        - DATA_CLASS_SMALL
    '''

    ##### Creating and saving Data Class - Without equal balnce ratings and stopwords
    # DataHandler4 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_WITH_STOP_WORDS')
    # DataHandler4.create_and_save_data()
    # DataHandler4.load_all_categories()
    # DataHandler4.saveDataHandlerClass('DATA_CLASS_WITH_STOP_WORDS') 

    # #### Creating and saving Data Class - Without equal balnce ratings and stopwords
    # DataHandler5 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_UNI_WITH_STOP_WORDS')
    # DataHandler5.create_and_save_data()
    # DataHandler5.load_all_categories()
    # DataHandler5.update_balance_ratings()
    # DataHandler5.saveDataHandlerClass('DATA_CLASS_UNI_WITH_STOP_WORDS') 

    # ##### Creating and saving Data Class - Without equal balnce ratings                     
    # DataHandler1 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS')   
    # DataHandler1.create_and_save_data()                                                  
    # DataHandler1.load_all_categories()                                                   
    # DataHandler1.update_remove_stop_words()                                              
    # DataHandler1.saveDataHandlerClass('DATA_CLASS')                                      

    # ##### Creating and saving Data Class
    # DataHandler2 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_UNI') 
    # DataHandler2.create_and_save_data()                                                
    # DataHandler2.load_all_categories()                                                 
    # DataHandler2.update_remove_stop_words()                                            
    # DataHandler2.update_balance_ratings()
    # DataHandler2.saveDataHandlerClass('DATA_CLASS_UNI')                                                

    # ##### Creating and saving small Data Class
    DataHandler3 = data_preprocessing.DataHandler(CATEGORIES[1:2], class_name='DATA_CLASS_SMALL') 
    DataHandler3.create_and_save_data()                                                
    DataHandler3.load_all_categories()                                                 
    DataHandler3.update_remove_stop_words()                                            
    DataHandler3.update_balance_ratings()
    DataHandler3.saveDataHandlerClass('DATA_CLASS_SMALL')
    
    ##### Summarize the data for the 3 classes
    # DataHandler1.summarize_data()
    # DataHandler2.summarize_data()                                                      
    # DataHandler3.summarize_data()

    return None


def execute_all_shallow_models(data_class):

    """
    This function is ued to execute,
    all shallow learning models,
    on a specific dataclass.
    All results is saved.
    """

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


   
# Save plot for given class
def save_plots(DataClass):
    vis = visualization.DataVisualizer(DataClass)
    vis.plot_review_ratio()
    vis.plot_avg_n_word_reveiw()
    vis.plot_common_words()
    vis.plot_n_sentence_length()


# Save plots for all data classes used in project
def save_all_plots():
    DataClassSmall = data_preprocessing.loadDataHandler('DATA_CLASS_SMALL') 
    DataClassUni = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')     
    DataClass = data_preprocessing.loadDataHandler('DATA_CLASS')            
    DataClassWithStopWords = data_preprocessing.loadDataHandler('DATA_CLASS_WITH_STOP_WORDS')
    DataClassUniWithStopWords = data_preprocessing.loadDataHandler('DATA_CLASS_WITH_STOP_WORDS')

    save_plots(DataClassSmall)
    save_plots(DataClassUni)
    save_plots(DataClass)
    save_plots(DataClassWithStopWords)
    save_plots(DataClassUniWithStopWords)


def train_with_wandb():

    """
    Initialize and traind with wandb.
    See the sweep config file.
    """

    data_class_name = 'DATA_CLASS_UNI'
    DATA_SMALL = data_preprocessing.loadDataHandler(data_class_name)
    DataClass = DATA_SMALL 

    # Initialize wandb for this training rucn
    run = wandb.init(project='1_TRY')  # Corrected project name
    config = run.config

    # Initialize and build the model using hyperparameters indb
    model = modelClass.Build_Models(DataClass,
                                    epochs_n=config.n_epochs)
    model.build_model()
    model.train()

    # wandb.save("modeltest.h5")
    # Finish the wandb run after training is complete
    run.finish()


# Save wandb (online) model locally
def wandb_save_model():
    # Initialize
    run = wandb.init(project="DATA_CLASS_UNI", job_type="model_loading")
    # Use the latest version of the artifact
    artifact = run.use_artifact('model-fallen-sweep-5:v9', type='model')
    artifact_dir = artifact.download() 
    

# Load (online) model, saved from WANDB
def wandb_load_model():
    """
    """
    wandb.init(project="DATA_CLASS_UNI", job_type="model_loading")
    model_builder = modelClass.Build_Models(DataClass)
    saved_model_path = './artifacts/model-fallen-sweep-5v9'
    model_builder.model = tf.keras.models.load_model(saved_model_path)
    return model_builder



if __name__ == "__main__":
    start_time = time.time()
    
    ##############################################    
    ##### Create and save the 2 data Classes #####
    #### Delete training, before call create #####
    ######## Remember to Delete files ############
    ##############################################

    create_and_save_data_classes(CATEGORIES)
    
    ##############################################
    ############## LOAD DATAHANDLER ##############
    ##############################################

    DataClassSmall = loadDataHandler('DATA_CLASS_SMALL') # Take 1 category
    DataClassUni = loadDataHandler('DATA_CLASS_UNI') # Json files 50mb For the 2 big ones
    DataClass = loadDataHandler('DATA_CLASS') # Json files 10mb for the 2 big ones 

    ##############################################
    ############## VISUALIZE #####################
    ##############################################

    # save_all_plots() # Save all plots for the 3 DataClasses

    ##############################################
    ##############   TRAIN  ######################
    ########### SHALLOW MODELS ###################
    ##############################################

    # execute_all_shallow_models(DataClassSmall)
    # execute_all_shallow_models(DataClassUni)
    # execute_all_shallow_models(DataClass)

    ##############################################
    ######## SHALLOW MODELS TEST RESULTS #########
    ##############################################

    # evaluate_all_shallow_models(DataClassUni)

    ############################################## 
    ########  ENSEMBLE SHALLOW MODELS    #########
    ############################################## 

    # ensemble_models = ['SVM', 'Boosting', 'DT']
    # ensemble_models = ['SVM', 'Boosting', 'RF']
    # ensemble_models = ['SVM', 'Boosting', 'RF', 'DT']
    # ensemble_predictions_majority_prioritize_SVM(DataClassUni, ensemble_models)

    ##### Evaluate Ensemble Model #####
    # ensemble_predictions_majority_prioritize_SVM_EVALUATE(DataClassUni, ensemble_models)

    ##############################################
    ########### Creating Sweep (RNN) ############# 
    ##############################################

    sweep_id = wandb.sweep(sweep_confi, project='TEST_3_FUN')
    wandb.agent(sweep_id, train_with_wandb, count=5)



    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
