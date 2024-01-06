import tensorflow as tf
import pandas as pd
import requests
import gzip
import io
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.simplefilter('ignore')

print("##### Downloading Data #####")

# To handle all the data
class DataHandler:
    def __init__(self, categories,
                 current_category = None,
                 chunk_size = 10000,
                 base_path = './Data/Training/',
                 json_path = './Data/Locally/'):
        self.base_path = base_path
        self.json_path = json_path
        self.chunk_size = chunk_size

        self.df_train = None
        self.df_val = None
        self.full_df = None
        self.categories = categories
        self.data_sets = {category: {'train': None, 'val': None, 'df': None} for category in categories}


    def load_full_data(self, specific_path):
        # Load from the json files 
        self.full_df = json_to_df(specific_path)
        self.full_df = dataFormatting(self.full_df)

    def load_data_for_category(self, category):
        if category not in self.categories:
            raise ValueError(f"Category '{category}' not in category list.")

        X_train = pd.read_csv(f"{self.base_path}{category}_X_train.csv")
        X_val = pd.read_csv(f"{self.base_path}{category}_X_val.csv")
        Y_train = pd.read_csv(f"{self.base_path}{category}_Y_train.csv")
        Y_val = pd.read_csv(f"{self.base_path}{category}_Y_val.csv")

        df_train = pd.concat([Y_train, X_train], axis=1)
        df_val = pd.concat([Y_val, X_val], axis=1)

        self._preprocess_data(df_train)
        self._preprocess_data(df_val)

        self.data_sets[category]['train'] = df_train
        self.data_sets[category]['val'] = df_val
    
    def load_all_categories(self):
        for category in self.categories:
            if category not in self.categories:
                raise ValueError(f"Category '{category}' not in category list.")
            
            # Load training data
            X_train = pd.read_csv(f"{self.base_path}{category}_X_train.csv")
            Y_train = pd.read_csv(f"{self.base_path}{category}_Y_train.csv")
            df_train = pd.concat([Y_train, X_train], axis=1)
            self._preprocess_data(df_train)

            # Load validation data
            X_val = pd.read_csv(f"{self.base_path}{category}_X_val.csv")
            Y_val = pd.read_csv(f"{self.base_path}{category}_Y_val.csv")
            df_val = pd.concat([Y_val, X_val], axis=1)
            self._preprocess_data(df_val)

            # Load df data
            print(f"{self.json_path}{category}_5.json")
            full_df_path = f"{self.json_path}{category}_5.json"
            full_df = json_to_df(full_df_path)


            # Store in the data_sets dictionary
            self.data_sets[category] = {'train': df_train, 'val': df_val, 'df': full_df}

    def _preprocess_data(self, df):
        df['overall'] = df['overall'] - 1  # Adjusting the 'overall' rating
        df['text'] = df['text'].fillna('').astype(str)


    def get_data(self, category, data_type):
        self.current_category = category
        return self.data_sets[category][data_type]

    def get_full_data(self):
        return self.full_df

    def get_current_cateogry(self):
        return self.current_category

    def get_list_of_categories(self):
        return self.categories


    # Save Datahanler
    def saveDataHandlerClass(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)



categories = ['Magazine_Subscriptions', 
            'Clothing_Shoes_and_jewelry', 
            'Arts_Crafts_and_Sewing', 
            'AMAZON_FASHION',
            'Luxury_Beauty']



def read_gzipped_json_from_url(url):
    # Send a HTTP request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Use gzip to decompress the content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            # Read the JSON lines file and convert to a DataFrame
            df = pd.read_json(gz, lines=True)
        return df
    else:
        print(f"Failed to retrieve data: status code {response.status_code}")
        return None



def dataFormatting(df):
    # Concatenating summary and reviewText
    df['text'] = df['summary'] + ' ' + df['reviewText']
    df['text'].fillna('', inplace=True)
    columns =  ['overall', 'text']
    new_df = df[columns]
    # Labels gets adjust in the RNN model

    # new_df = new_df.sample(n=n, random_state=42)
    return new_df 



# Save Categories as RAW DF 
def saveCategoriesDF(l):
    for item in l:
        print(f"##### Downloading: {item} #####")
        url = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/' + item + '_5.json.gz'
        df = read_gzipped_json_from_url(url).reset_index(drop=True)
        if df is not None:
            df.to_csv('./Data/Raw/Raw_' + item + '.csv', index=False)



# Split DF and locally save Training
def saveLocallyTraining(df, name):
    
    y = df['overall']
    df_small = df[[ 'text']]
    path_save = './Data/Training/'


    # Split the data into training and testing sets
    df_X_train, df_Xtest, df_ytrain, df_ytest = train_test_split(df_small, y, test_size=0.1, random_state=42, stratify=y)

    df_X_train = df_X_train.reset_index(drop=True)
    df_X_val = df_Xtest.reset_index(drop=True)
    df_y_train = df_ytrain.reset_index(drop=True)
    df_y_val = df_ytest.reset_index(drop=True)

    df_X_train.to_csv(path_save + name + '_X_train.csv', index=False)
    df_X_val.to_csv(path_save + name + '_X_val.csv', index=False)
    df_y_train.to_csv(path_save + name + '_Y_train.csv', index=False)
    df_y_val.to_csv(path_save + name + '_Y_val.csv', index=False)



# Split and Save 
def df_read_save_csv(l):
    for item in l:
        df = pd.read_csv('./Data/Raw/Raw_' + item + '.csv', chunksize=10000)
        df = dataFormatting(df)
        saveLocallyTraining(df, item)


def df_read_save_json(l):
    chunk_size = 10000
    path = './Data/Locally/'
    for item in l:
        json_path = path + item + '_5.json'
        json_reader = pd.read_json(json_path, lines = True, chunksize = chunk_size)
        for chunk in json_reader:
            chunk = dataFormatting(chunk)
            saveLocallyTraining(chunk, item)
        # df = dataFormatting(df)
        # saveLocallyTraining(df, item)

def df_read_save_json2(l):
    for item in l:
        path = './Data/Locally/'
        df = pd.read_json(path + item + '_5.json', lines = True, chunksize=10000)
        df = dataFormatting(df)
        saveLocallyTraining(df, item)




def sample_df(df, n):
    return df.sample(n=n, random_state=42)


def json_to_df(category_path):
    return pd.read_json(category_path, lines = True)



# This is if the files were saved manually through links
class DataloaderSavedLocally():
    def __init__(self, categories = ['Clothing_Shoes_and_jewelry']): # categories = ['Magazine_Subscriptions', 
                       #               'Clothing_Shoes_and_jewelry', 
                       #               'Arts_Crafts_and_Sewing', 
                       #               'AMAZON_FASHION',
                       #               'Luxury_Beauty']):
        df_read_save_json(categories)




# Load data_handler class 
def loadDataHandler(class_path):
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        return data_handler








# categories = ['Magazine_Subscriptions', 'Clothing_Shoes_and_jewelry', 'AMAZON_FASHION', 'Luxury_Beauty']
# categories = ['Magazine_Subscriptions', 'Clothing_Shoes_and_jewelry', 'Arts_Crafts_and_Sewing', 'AMAZON_FASHION', 'Luxury_Beauty']



# ALreaydy saved locally
# df_read_save_json(categories)

##### Possibly use chunk size #####
# https://towardsai.net/p/data-science/efficient-pandas-using-chunksize-for-large-data-sets-c66bf3037f93

##### Links to download #####
#https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_jewelry_5.json.gz
# https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz
#https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
#https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
#https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz



# Old datahandler saved 
class DataHandler2:
    def __init__(self, item, categories,
                 base_path = './Data/Locally/'):
        self.base_path = base_path
        self.item = item
        self.df_train = None
        self.df_val = None
        self.full_df = None
        self.categories = categories
        self.data_sets = {category: {'train': None, 'val': None} for category in categories}

    def load_training_validation_data(self):
        X_train = pd.read_csv(f"{self.base_path}{self.item}_X_train.csv")
        X_val = pd.read_csv(f"{self.base_path}{self.item}_X_val.csv")
        Y_train = pd.read_csv(f"{self.base_path}{self.item}_Y_train.csv")
        Y_val = pd.read_csv(f"{self.base_path}{self.item}_Y_val.csv")

        self.df_train = pd.concat([Y_train, X_train], axis=1)
        self.df_val = pd.concat([Y_val, X_val], axis=1)

        self._preprocess_data(self.df_train)
        self._preprocess_data(self.df_val)

    def load_full_data(self, specific_path):
        # Load from the json files 
        self.full_df = json_to_df(specific_path)
        self.full_df = dataFormatting(self.full_df)

    def load_data_for_category(self, category):
        if category not in self.categories:
            raise ValueError(f"Category '{category}' not in category list.")

        X_train = pd.read_csv(f"{self.base_path}{category}_X_train.csv")
        X_val = pd.read_csv(f"{self.base_path}{category}_X_val.csv")
        Y_train = pd.read_csv(f"{self.base_path}{category}_Y_train.csv")
        Y_val = pd.read_csv(f"{self.base_path}{category}_Y_val.csv")

        df_train = pd.concat([Y_train, X_train], axis=1)
        df_val = pd.concat([Y_val, X_val], axis=1)

        self._preprocess_data(df_train)
        self._preprocess_data(df_val)

        self.data_sets[category]['train'] = df_train
        self.data_sets[category]['val'] = df_val

    def _preprocess_data(self, df):
        df['overall'] = df['overall'] - 1  # Adjusting the 'overall' rating
        df['text'] = df['text'].fillna('').astype(str)

    def get_training_data(self):
        return self.df_train

    def get_validation_data(self):
        return self.df_val

    def get_full_data(self):
        return self.full_df


    # Save Datahanler
    def saveDataHandler(self, file_name):
        with open('data_handler.pkl', 'wb') as output:
          pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)



