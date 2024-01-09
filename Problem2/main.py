# Import our files
from tensorflow._api.v2 import train
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
project_name = 'mytestporject' 
entity = 'marcs20'             
# wandb.login()
# wandb.init(project=project_name, entity=entity)
#############################


sweep_config = {
        'method': 'random',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
           'name': 'val_accuracy',
           'goal': 'maximize'   
        },
        'parameters': {
            'batch_size': {
                'values': [64, 64, 64]
            },
            'embedding_dimension': {
                'values': [1000, 1000, 1000]
            },
            'n_epochs': {
                'values': [5, 5, 5]
            },
        }
}

CATEGORIES = ['This_is_bull1',
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # Er ikke downloaded endnu 
              'All_Beauty',
              'This_is_bull2',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
              # 'Luxury_Beauty'] # Test Data


def train_with_wandb():
    quit()


def create_load_and_save_data_class(CATEGORIES):
    DataHandler = data_preprocessing.DataHandler(CATEGORIES)
    DataHandler.get_Data_From_Local_Json() # Remember to delete train folder
    DataHandler.load_all_categories()
    DataHandler.combine_data()
    DataHandler.saveDataHandlerClass('Data_Class')



if __name__ == "__main__":
    start_time = time.time()
    ##### Create- and save the DataClass #####
    # create_load_and_save_data_class(CATEGORIES) # Time 205 Sec
         
    DataClass = data_preprocessing.loadDataHandler('Data_Class')
    DataHandler: data_preprocessing.DataHandler = DataClass # Loading DataHandler class
    DataVisualizer = visualization.DataVisualizer(DataHandler) # Creating Visualization Class

    # combined_train = DataHandler.get_combined_train_data()


    ##### Visiualize Data #####  
    DataVisualizer.plot_common_words()
    DataVisualizer.plot_avg_n_word_reveiw()
    DataHandler.remove_stop_words_in_combined() # Remove stop words in combined
    DataVisualizer.plot_avg_n_word_reveiw()
    DataVisualizer.plot_common_words()
    DataVisualizer.plot_n_sentence_length()
    DataVisualizer.plot_review_ratio()




    #####  Creating the model ####
    # model = modelClass.Build_Models(DataHandler, epochs_n=3)
    # model.build_model()
    # model.train()

     





    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")


    


##### Notes: #####
## Before remove stop words
# Batch size = 285 

