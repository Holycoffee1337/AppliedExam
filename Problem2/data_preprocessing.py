'''
File: data_preprocessing.py
Purpose: provides DataHandler class,
that handle all the data.
'''
import tensorflow as tf
import pandas as pd
import requests
import gzip
import io
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import pickle
import os
import nltk
from nltk.corpus import stopwords
warnings.simplefilter('ignore')

# To handle all the data
class DataHandler:
    def __init__(self, categories,
                 current_category = None,
                 chunk_size = 300,
                 base_path = './Data/Training/',
                 json_path = './Data/Locally/'):
        self.base_path = base_path
        self.json_path = json_path
        self.chunk_size = chunk_size

        self.df_train = None
        self.df_val = None
        self.full_df = None
        self.test_data = None
        self.load_test_data()

        self.combined_train_data = None
        self.combined_val_data = None
        self.categories = categories
        self.data_sets = {category: {'train': None, 'val': None, 'df': None} for category in categories}

    def get_Data_From_Local_Json(self): 
        print('##### Creating DF-, Train- and Validation data #####')
        valid_categories = df_read_save_json(self.categories)
        self.categories = valid_categories

    def load_test_data(self):
        category = 'Luxury_Beauty'
        full_df_path = f"{self.json_path}{category}_5.json"
        full_df = json_to_df(full_df_path)
        full_df = dataFormatting(full_df)
        self.test_data = full_df


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
        print('##### Load all data into class #####')
        print('All training and validation data is loaded into class.')
        print(f'For the categories: {self.categories}')

        for category in self.categories:
            def process_chunks(file_path):
                i = 0 
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    i = i + 1
                    # print(i)
                    # You can add any preprocessing here if needed
                    chunks.append(chunk)
                return pd.concat(chunks, axis=0) # axis = 0??

            # Load training data
            X_train = process_chunks(f"{self.base_path}{category}_X_train.csv")
            Y_train = process_chunks(f"{self.base_path}{category}_Y_train.csv")
            df_train = pd.concat([Y_train, X_train], axis=1)
            self._preprocess_data(df_train)  # Preprocess the concatenated DataFrame

            # Load validation data
            X_val = process_chunks(f"{self.base_path}{category}_X_val.csv")
            Y_val = process_chunks(f"{self.base_path}{category}_Y_val.csv")
            df_val = pd.concat([Y_val, X_val], axis=1)
            self._preprocess_data(df_val)  # Preprocess the concatenated DataFrame

            # Load df data
            # print(f"{self.json_path}{category}_5.json")
            full_df_path = f"{self.json_path}{category}_5.json"
            full_df = json_to_df(full_df_path)
            full_df = dataFormatting(full_df)

            # Store in the data_sets dictionary
            self.data_sets[category] = {'train': df_train, 'val': df_val, 'df': full_df}

        print('')
        print('')


    def combine_data(self):
        all_train_data = []
        all_val_data = []

        for category in self.data_sets:
            if self.data_sets[category]['train'] is not None:
                all_train_data.append(self.data_sets[category]['train'])

            if self.data_sets[category]['val'] is not None:
                all_val_data.append(self.data_sets[category]['val'])

        # Concatenate all training data
        combined_train_data = pd.concat(all_train_data, ignore_index=True) if all_train_data else None

        # Concatenate all validation data
        combined_val_data = pd.concat(all_val_data, ignore_index=True) if all_val_data else None

        # Shuffle all the data 
        combined_train_data_shuffled = combined_train_data.sample(frac=0.1, random_state=42).reset_index(drop=True)
        combined_val_data_shuffled = combined_val_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

        self.combined_train_data = combined_train_data_shuffled
        self.combined_val_data = combined_val_data_shuffled

    def remove_stop_words_in_combined(self):
        self.combined_train_data = remove_stop_words(self.combined_train_data)
        self.combined_val_data = remove_stop_words(self.combined_val_data)
        

    def _preprocess_data(self, df):
        df['overall'] = df['overall'] - 1  # Adjusting the 'overall' rating
        df['text'] = df['text'].fillna('').astype(str)
        # print(df['text'])


    def get_data(self, category, data_type):
        self.current_category = category
        return self.data_sets[category][data_type]

    def get_full_data(self):
        return self.full_df

    def get_current_cateogry(self):
        return self.current_category

    def get_list_of_categories(self):
        return self.categories


    def get_combined_train_data(self):
        return self.combined_train_data

    def get_combined_val_data(self):
        return self.combined_val_data

    def get_test_data(self):
        return self.test_data

    # Save Datahanler
    def saveDataHandlerClass(self, file_name):
        print(f'DataHandler class saved as: {file_name}')
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)



def remove_stop_words(df):
    stop_words = set(stopwords.words('english')) 
    filtered_text = []
    for text in df['text']:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(filtered_words))
    df['text'] = filtered_text
    return df 
 

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
    df_small = df[['text']]
    path_save = './Data/Training/'

    # # Define file paths
    # file_X_train = path_save + name + '_X_train.csv'
    # file_X_val = path_save + name + '_X_val.csv'
    # file_y_train = path_save + name + '_Y_train.csv'
    # file_y_val = path_save + name + '_Y_val.csv'
    # 
    # # Remove existing files if they exist to start fresh
    # for file in [file_X_train, file_X_val, file_y_train, file_y_val]:
    #     if os.path.exists(file):
    #         os.remove(file)

    # Split the data into training and validation sets
    df_X_train, df_X_val, df_y_train, df_y_val = train_test_split(df_small, y, test_size=0.1, random_state=42, stratify=y)

    # Reset index for all splits
    df_X_train = df_X_train.reset_index(drop=True)
    df_X_val = df_X_val.reset_index(drop=True)
    df_y_train = df_y_train.reset_index(drop=True)
    df_y_val = df_y_val.reset_index(drop=True)

    # Define the mode for opening the file, 'a' for append and 'w' for write
    mode_X_train = 'a' if os.path.exists(path_save + name + '_X_train.csv') else 'w'
    mode_X_val = 'a' if os.path.exists(path_save + name + '_X_val.csv') else 'w'
    mode_y_train = 'a' if os.path.exists(path_save + name + '_Y_train.csv') else 'w'
    mode_y_val = 'a' if os.path.exists(path_save + name + '_Y_val.csv') else 'w'

    # Write the data to CSV files, append if file exists, otherwise write
    df_X_train.to_csv(path_save + name + '_X_train.csv', mode=mode_X_train, index=False, header=not os.path.exists(path_save + name + '_X_train.csv'))
    df_X_val.to_csv(path_save + name + '_X_val.csv', mode=mode_X_val, index=False, header=not os.path.exists(path_save + name + '_X_val.csv'))
    df_y_train.to_csv(path_save + name + '_Y_train.csv', mode=mode_y_train, index=False, header=not os.path.exists(path_save + name + '_Y_train.csv'))
    df_y_val.to_csv(path_save + name + '_Y_val.csv', mode=mode_y_val, index=False, header=not os.path.exists(path_save + name + '_Y_val.csv'))



# Split and Save 
def df_read_save_csv(l):
    for item in l:
        df = pd.read_csv('./Data/Raw/Raw_' + item + '.csv', chunksize=10000)
        df = dataFormatting(df)
        saveLocallyTraining(df, item)


def df_read_save_json(l):
    chunk_size = 1000 # 10000
    path = './Data/Locally/'
    valid_categories = []
    for item in l:
        json_path = path + item + '_5.json'

        if not os.path.exists(json_path):
             print('')
             print(f"File {json_path} does not exist. Skipping...")
             print('Download the json file (category) manually, and put in ./Data/Locally/')
             print('')
             continue
        print(f'Downloaded: {item}')
        valid_categories.append(item)
        json_reader = pd.read_json(json_path, lines = True, chunksize = chunk_size)
        for chunk in json_reader:
            chunk = dataFormatting(chunk)
            saveLocallyTraining(chunk, item)
        # df = dataFormatting(df)
        # saveLocallyTraining(df, item)
    print('Categories do now have train and val data saved')
    print('')
    print('')
    return valid_categories



def sample_df(df, n):
    return df.sample(n=n, random_state=42)


def sample_fixed_number_by_group(df, group_column, n_samples_per_group):
    min_group_size = df[group_column].value_counts().min()
    n_samples_per_group = min(n_samples_per_group, min_group_size)
    
    sampled_df = df.groupby(group_column).apply(
        lambda x: x.sample(n=n_samples_per_group) if len(x) >= n_samples_per_group else x
    ).reset_index(drop=True)
    return sampled_df



def json_to_df(category_path):
    return pd.read_json(category_path, lines = True)



# Load data_handler class 
def loadDataHandler(class_path):
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        return data_handler




# Add -> And check how big the file is
# Implement: -> So it can take big json files -> Create smaller json files





# Reduce the json fiels
def reduce_json_files(max_size_gb=0.2):
    original_file_path = './Data/Locally/Clothing_Shoes_and_jewelry_5.json'
    new_file_path = './Data/SmallJsonFiles/Clothing_Shoes_and_jewelry_5.json'

    # Check the size of the original file
    file_size_bytes = os.path.getsize(original_file_path)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)  # Convert bytes to gigabytes

    # If the file is 1GB or larger, create a new file with only the first 1GB of data
    if file_size_gb >= max_size_gb:
        with open(original_file_path, 'r', encoding='utf-8') as original_file, \
             open(new_file_path, 'w', encoding='utf-8') as new_file:
            
            bytes_written = 0
            max_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes

            for line in original_file:
                # Check if adding the next line exceeds the maximum size
                if bytes_written + len(line.encode('utf-8')) > max_bytes:
                    break
                new_file.write(line)
                bytes_written += len(line.encode('utf-8'))

        print(f"Created a smaller file: {new_file_path}")
    else:
        print(f"The file is smaller than {max_size_gb}GB. No new file created.")


# reduce_json_files()
# Example usage
# original_file = 'path/to/your/large_file.json'
# new_file = 'path/to/your/smaller_file.json'
# create_smaller_json_if_large(original_file, new_file)
