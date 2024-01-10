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
project_name = 'Chris_Test' 
entity = 'marcs20'             
wandb.login()
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

CATEGORIES = ['This_is_bull1',
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # Er ikke downloaded endnu 
              'All_Beauty',
              'This_is_bull2',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
              # 'Luxury_Beauty'] # Test Data


def train_with_wandb():
    # Load Data Handler
    DataClass = data_preprocessing.loadDataHandler('Data_Class_2')
    DataHandler: data_preprocessing.DataHandler = DataClass
    DataHandler.remove_stop_words_in_combined()

    # Initialize wandb for this training run
    run = wandb.init(project='mytestproject')  # Corrected project name
    config = run.config

    # Initialize and build the model using hyperparameters from wandb
    model = modelClass.Build_Models(DataHandler, 
                                    epochs_n=config.n_epochs,
                                    batch_size=config.batch_size)
    model.build_model()
    model.train()

    # Finish the wandb run after training is complete
    run.finish()


def create_load_and_save_data_class(CATEGORIES):
    DataHandler = data_preprocessing.DataHandler(CATEGORIES)
    DataHandler.get_Data_From_Local_Json() # Remember to delete train folder
    DataHandler.load_all_categories()
    DataHandler.combine_data()
    DataHandler.saveDataHandlerClass('Data_Class_2')


def train():
    with wandb.init() as run:
        config = run.config
        data = preprocess_data()  # Your data preprocessing
        model = MyModel(learning_rate=config.learning_rate, num_layers=config.num_layers)
        model.train(batch_size=config.batch_size)


if __name__ == "__main__":
    start_time = time.time()


    ##### Creating Sweep Model #####
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train_with_wandb)

     
    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
