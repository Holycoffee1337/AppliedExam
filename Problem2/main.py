# Import our files
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
import warnings
warnings.filterwarnings('ignore')

# Our Files
import data_preprocessing
import visualization

##### Initialize wandb #####
os.environ["WANDB_API_KEY"] = '76fdf9acb6a4a334b5b8c8f89c8a63c1c2b5135e'
project_name = 'Chris_Test' 
entity = 'marcs20'             
wandb.login()
############################


sweep_config = {
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
           'name': 'val_accuracy',
           'goal': 'maximize'   
        },
        'parameters': { # output_sequence_length: Choose from plot
            'batch_size': {
                'values': [128, 224]
                # },
                # 'embedding_dimension': {
                #     'values': [200, 400, 1000]
            },
            'n_epochs': {
                'values': [1]
            },
        }
}

CATEGORIES = [
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # Er ikke downloaded endnu 
              'All_Beauty',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
              # 'Luxury_Beauty'] # Test Data


def train_with_wandb():
    # Load Data Handler
    DataClass = data_preprocessing.loadDataHandler('Data_Class_2')
    DataHandler: data_preprocessing.DataHandler = DataClass

    # Initialize wandb for this training rucn
    run = wandb.init(project='mytestproject')  # Corrected project name
    config = run.config

    # Initialize and build the model using hyperparameters indb
    model = modelClass.Build_Models(DataHandler, 
                                    epochs_n=config.n_epochs,
                                    batch_size=config.batch_size)
    model.build_model()
    model.train()

    # Finish the wandb run after training is complete
    run.finish()


def create_and_save_data_class(CATEGORIES, name):
    DataHandler = data_preprocessing.DataHandler(CATEGORIES)
    DataHandler.create_and_save_data()
    # DataHandler.load_category('AMAZON_FASHION')
    DataHandler.load_all_categories()


    # 
    DataHandler.saveDataHandlerClass('MY_TEST') 
    DataHandler.update_remove_stop_words()
    # DataHandler.update_balance_ratings()
    DataHandler.saveDataHandlerClass(name) 
    return DataHandler



def load_and_visualize(name):
    DataClass: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler(name)
    DataVisualizer = visualization.DataVisualizer(DataClass)
    # DataClass.update_remove_stop_words()
    # DataClass.update_balance_ratings()
    # print(DataClass.get_test_data())
    

    # Plots
    DataVisualizer.plot_avg_n_word_reveiw()
    DataVisualizer.plot_review_ratio()





if __name__ == "__main__":
    start_time = time.time()
    
    ##############################################
    ##### Create and save the 2 data Classes #####
    #### Delete training, before call create #####
    ######## Remember to Delete files ############
    ##############################################

    create_and_save_data_class(CATEGORIES, 'TEST')

    ##############################################
    ############## LOAD CLASS ####################
    ##############################################
    # load_and_visualize('DATA_CLASS')

    ##############################################
    ########### Creating Sweep Model ############# 
    ##############################################

    # sweep_id = wandb.sweep(sweep_config, project=project_name)
    # wandb.agent(sweep_id, train_with_wandb)

     
    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
