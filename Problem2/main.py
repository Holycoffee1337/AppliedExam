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
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')

# Our Files
import data_preprocessing
from data_preprocessing import DataHandler, loadDataHandler
import visualization

 #import shallow_learning

##### Initialize wandb #####
os.environ["WANDB_API_KEY"] = '76fdf9acb6a4a334b5b8c8f89c8a63c1c2b5135e'
project_name = 'Chris_Test' 
entity = 'marcs20'             
wandb.login()
# Set wandb to offline mode
os.environ['WANDB_MODE'] = 'dryrun'
############################



def create_and_save_data_class(CATEGORIES, name, balance_ratings):

    DataHandler = data_preprocessing.DataHandler(CATEGORIES, class_name=name)
    DataHandler.create_and_save_data()
    DataHandler.load_all_categories()

    DataHandler.update_remove_stop_words()
    if balance_ratings == True:
        DataHandler.update_balance_ratings() 

    DataHandler.saveDataHandlerClass(name) 
    DataHandler.summarize_data()

    return DataHandler


def create_and_save_data_classes(CATEGORIES):

    # Creating and saving Data Class - Without equal balnce ratings and stopwords
    DataHandler = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_WITH_STOP_WORDS')
    DataHandler.create_and_save_data()
    DataHandler.load_all_categories()
    DataHandler.saveDataHandlerClass('DATA_CLASS_WITHOUT_STOP_WORDS') 

    # Creating and saving Data Class - Without equal balnce ratings                     
    DataHandler1 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS')   
    DataHandler1.create_and_save_data()                                                  
    DataHandler1.load_all_categories()                                                   
    DataHandler1.update_remove_stop_words()                                              
    DataHandler1.saveDataHandlerClass('DATA_CLASS')                                      

    # Creating and saving Data Class
    DataHandler2 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_UNI') 
    DataHandler2.create_and_save_data()                                                
    DataHandler2.load_all_categories()                                                 
    DataHandler2.update_remove_stop_words()                                            
    DataHandler2.update_balance_ratings()
    DataHandler2.saveDataHandlerClass('DATA_CLASS_UNI')                                                

    # Creating and saving small Data Class
    DataHandler3 = data_preprocessing.DataHandler(CATEGORIES[:1], class_name='DATA_CLASS_SMALL') 
    DataHandler3.create_and_save_data()                                                
    DataHandler3.load_all_categories()                                                 
    DataHandler3.update_remove_stop_words()                                            
    DataHandler3.update_balance_ratings()
    DataHandler3.saveDataHandlerClass('DATA_CLASS_SMALL')
    
    # Summarize the data for the 3 classes
    DataHandler1.summarize_data()
    DataHandler2.summarize_data()                                                      
    DataHandler3.summarize_data()

    return None


def save_visualize(DataClass: DataHandler):
    path_folder = './RESULTS/PLOTS/'
    class_name = DataClass.get_class_name
    DataVisualizer = visualization.DataVisualizer(DataClass) 
    print(DataClass.get_list_of_categories())

    DataVisualizer.plot_avg_n_word_reveiw()
    # plt.savefig(f'{path_folder}{class_name}_avg_word_review_plot.png')


    # DataVisualizer.plot_avg_n_word_reveiw()
    




def load_and_visualize(name):
    quit()
    # DataClass: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler(name)
    # DataVisualizer = visualization.DataVisualizer(DataClass)
    # DataClass.update_remove_stop_words()
    # DataClass.update_balance_ratings()
    # print(DataClass.get_test_data())


    # DataClass.summarize_data()
    # print(DataClass.get_class_name)
    

    # Plots: Categories
    # DataVisualizer.plot_avg_n_word_reveiw()
    # DataVisualizer.plot_review_ratio()

    # Plots: Category Clothing_Shoes_and_jewelry (Biggest)
    # DataVisualizer.plot_common_words('Clothing_Shoes_and_jewelry')
    # DataVisualizer.plot_n_sentence_length('Clothing_Shoes_and_jewelry')


def train_with_wandb():
    # Load Data Handler
    # DATA: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS')
    # DATA_UNI: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')
    # DATA: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS')
    data_class_name = 'DATA_CLASS_SMALL_UNI'
    print(f'##### {data_class_name} #####')
    DATA_SMALL = data_preprocessing.loadDataHandler(data_class_name)
    DataClass = DATA_SMALL 

    # Initialize wandb for this training rucn
    run = wandb.init(project='1_TRY')  # Corrected project name
    config = run.config

    # Initialize and build the model using hyperparameters indb
    model = modelClass.Build_Models(DataClass,
                                    epochs_n=config.n_epochs)
                                    # batch_size=config.batch_size)
    model.build_model()
    model.train()
    # model.evaluate()

    # Finish the wandb run after training is complete
    run.finish()



sweep_config = {
        'method': 'random',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
           'name': 'val_accuracy',
           'goal': 'maximize'   
        },
        'parameters': { # output_sequence_length: Choose from plot
                       #'batch_size': {
                       #'values': [128, 224]
                # },
                # 'embedding_dimension': {
                #     'values': [200, 400, 1000]
            
            'n_epochs': {
                'values': [5]
            },

            'lr': {
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            'output_sequence_length':{
                'values': [25, 100]
            },
            'embedding_dimension':{
                'values': [200, 400]
            },

                       # embedding_dimension=200,     # 
                

                       # output_sequence_length=200,   # 75% Fractil: 90% Fractil

        }
}

# Categories we want to use for our dataset
CATEGORIES = [
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # BIGGEST 
              'All_Beauty',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
 

if __name__ == "__main__":
    start_time = time.time()
    CLASS_FOLDER_PATH = './RESULTS/DATA_CLASSES/'
    
    ##############################################
    ##### Create and save the 2 data Classes #####
    #### Delete training, before call create #####
    ######## Remember to Delete files ############
    ##############################################

    create_and_save_data_classes(CATEGORIES)

    ##############################################
    ############## VISUALIZE #####################
    ##############################################

    # DataHandler = loadDataHandler(CLASS_FOLDER_PATH + 'DATA_CLASS') 
    # save_visualize(DataHandler)




    ##############################################
    ############## LOAD CLASS ####################
    ##############################################

    # DATA: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS')
    # DATA_UNI: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')
    # DATA_SMALL = data_preprocessing.loadDataHandler('DATA_CLASS_SMALL')

    # DATA = data_preprocessing.loadDataHandler('DATA_CLASS_SMALL_UNI_TRAIN')
    # DATA.summarize_data()


    # DATA.summarize_data()
    # DATA_UNI.summarize_data()


    ##############################################
    ########### Creating Sweep Model ############# 
    ##############################################

    # sweep_id = wandb.sweep(sweep_config, project=project_name)
    # wandb.agent(sweep_id, train_with_wandb, count=5)

     
    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
