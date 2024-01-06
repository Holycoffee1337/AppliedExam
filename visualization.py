'''
File: visualization.py
Purpose: Provide functions for visualization.
'''
import pandas as pd
from pandas._libs.hashtable import value_count
import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
# Own Files imported
import data_preprocessing 



class DataVisualizer:
    def __init__(self, DataHandler):
        self.DataHandler = DataHandler

    # Plots for list of categories
    def plot_avg_sentences(self):
        quit()
        # cat_avg_sentence(self.l, self.path_folder)

    def plot_review_ratio(self):
        """
        Calculate the procent for each key, given the total values overall.

        Parameters:
           l (list): List containing names of categories 
           path: path to the locally folder containing .json files for each category 

        Returns:
            None

        Execute:
            Show plot of categories and there % ratings

        """
        reviewRatio(self.DataHandler)


    # Plots for specific Category
    def plot_common_words(self, category):
        """
        Analyzes the text data in the 'text' column of the provided DataFrame and
        prints the top 10 most common words along with their frequencies.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

        Returns:
            None

        Execution: 
            Show plot
        """
        df = self.DataHandler.get_data(category, 'df')
        common_words(df, category)

    def plot_n_sentence_length(self, category_name):
        """
        Find the length of all sentences for the category chosen, 
        and make a histogram over it. To visualize, how long
        each sentence is.
    
        Parameters:
            cateogry_name: The name for the category, you want to plot 
    
        Returns:
            None
    
        Execution: 
            Show histogram plot
        """
    
        df = self.DataHandler.get_data(category_name, 'df')
        n_sentence_length(df, category_name)





def count_words(text):
    if pd.isna(text):  # Check for NaN values
        return 0
    words = str(text).split()  # Convert to string and split
    return len(words)

# Apply the function to each row and create a new column 'Word_Count'


def n_sentence_length(df, name):
    df['Word_Count'] = df['text'].apply(count_words)
    # print(df['Word_Count'].mean())

    percentiles = df['Word_Count'].describe(percentiles=[0.25, 0.5, 0.75])


    # Display the DataFrame with the new 'Word_Count' column
    # print(df[['text', 'Word_Count']])
    
    # Create a dictionary to store the count of rows for each word count
    word_count_dict = {}
    for count in df['Word_Count']:
        if count in word_count_dict:
            word_count_dict[count] += 1
        else:
            word_count_dict[count] = 1
    
    # Convert the dictionary items to lists for plotting
    word_counts = list(word_count_dict.keys())
    row_counts = list(word_count_dict.values())

    # Create fig
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 8))

    table_data = [['25%', f"{percentiles['25%']:.2f}"],
                ['50%', f"{percentiles['50%']:.2f}"],
                ['75%', f"{percentiles['75%']:.2f}"]]
    table = ax2.table(cellText=table_data, loc='center', 
                     colLabels=['Percentile', 'Value'], cellLoc='center', colColours=['#f0f0f0']*2)


    ax2.axis('off')
    # Plotting the bar chart
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Number of Rows')
    ax1.set_title(name + ': Word Count Distribution in Rows')
    ax1.bar(word_counts, row_counts, color='blue')
    # ax1.xlabel('Word Count')
    # ax1.ylabel('Number of Rows')
    # ax1.title('Word Count Distribution in Rows')
    plt.show()
    


def cat_avg_sentence(l, path_folder):
    dict_list = []
    for item in l:
        item_path = path_folder + item + '_5.json'
        df = data_preprocessing.json_to_df(item_path)
        df = data_preprocessing.dataFormatting(df)
        df['Word_Count'] = df['text'].apply(count_words)
        # dict_list.append(df['overall'].value_counts().to_dict())
        # dict_list.append(df['Word_Count'].mean().to_dict())
        dict_list.append({item: df['Word_Count'].mean()})


    categories = [list(d.keys())[0] for d in dict_list]
    avg_words = [list(d.values())[0] for d in dict_list]
    plt.bar(categories, avg_words, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Average Words')
    plt.title('Average N Words for Different Categories')
    plt.show()





def calcProcent(dic): 
    """
    Calculate the procent for each key, given the total values overall.

    Parameters:
        Dicitionary: The Dictionary containing the ratings as keys, and total as values 

    Returns:
        array: Where each position correlates to the key - 1

    """
    procent_list = []
    total_reviews = dic[1] + dic[2] + dic[3] + dic[4] + dic[5]
    i = 1
    total_procent = 0
    while i < 6:
        procent = (100 * (dic[i] / total_reviews))
        procent_list.append(procent)
        i = i + 1
        total_procent = procent + total_procent
        
    print('Total Procent')
    print(total_procent)
    return procent_list




def reviewRatio(DataHandler):
    """
    Calculate the procent for each key, given the total values overall.

    Parameters:
       l (list): List containing names of categories 
       path: path to the locally folder containing .json files for each category 

    Returns:
        None

    Execute:
        Show plot of categories and there % ratings

    """


    dict_list = []
    categories = DataHandler.get_list_of_categories()
    for category in categories:
        df = DataHandler.get_data(category, 'df')
        dict_list.append(df['overall'].value_counts().to_dict())

        
    One = []
    Two = []
    Three = []
    Four = [] 
    Five = []
    
    for item in dict_list:
        array_procent = calcProcent(item)
        ii = 0 
        while ii < len(array_procent):
            if ii == 0:
                One.append(array_procent[ii])
            if ii == 1:
                Two.append(array_procent[ii])
            if ii == 2:
                Three.append(array_procent[ii])
            if ii == 3:
                Four.append(array_procent[ii])
            if ii == 4:
                Five.append(array_procent[ii])
                
            ii = ii + 1


    # set width of bar 
    barWidth = 0.15
    fig = plt.subplots(figsize =(12, 8)) 
     

    # Set position of bar on X axis 
    br1 = np.arange(len(One)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 
     
    # Make the plot
    plt.bar(br1, One, color ='r', width = barWidth, 
            edgecolor ='grey', label ='One') 
    plt.bar(br2, Two, color ='g', width = barWidth, 
            edgecolor ='grey', label ='Two') 
    plt.bar(br3, Three, color ='b', width = barWidth, 
            edgecolor ='grey', label ='Three') 
    plt.bar(br4, Four, color ='black', width = barWidth, 
            edgecolor ='grey', label ='Four') 
    plt.bar(br5, Five, color ='purple', width = barWidth, 
            edgecolor ='grey', label ='Five') 
     

    # Adding Xticks 
    plt.xlabel('Categories', fontweight ='bold', fontsize = 15) 
    plt.ylabel('N Reviews', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(len(One))], 
            l)
     
    plt.legend()
    plt.show()
    return None





def common_words(df_format, category_name):
    """
    Analyzes the text data in the 'text' column of the provided DataFrame and
    prints the top 10 most common words along with their frequencies.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

    Returns:
        None

    Execution: 
        Show plot
    """

    # Handle NaN values by replacing them with an empty string
    df_format['text'] = df['text'].replace(np.nan, '', regex=True)
    # Use CountVectorizer to tokenize and count word frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each word across all sentences
    word_frequencies = Counter(dict(zip(feature_names, X.sum(axis=0).A1)))
    print(word_frequencies['positive'])
    # Remove specific keys
    # word_frequencies.pop('positive', None)
    # word_frequencies.pop('negative', None)
    
    # Display the most common words and their frequencies
    most_common_words = word_frequencies.most_common(10)
    most_common_words = most_common_words[0:10]
    #for word, frequency in most_common_words:
        # print(f"{word}: {frequency}")
    
    # Plot a bar chart of word frequencies
    plt.bar(*zip(*most_common_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(category_name + ': ' + 'Top 10 Most Common Words')
    plt.show()





###################################
############# TEST ################
#### Of different functions #######
###################################


###################
#### Categories ###
###################

###### Cat Avg Sentences ######
# cat_avg_sentence(l, path_folder)
# 
# 
# ##### Review Ratio  #########
# reviewRatio(l, path_folder)
# 
# 
# 
# ###################
# #### Individual ###
# ###################
# 
# ###### Common Words ######
# common_words(df)
# 
# 
# ##### Setence Length #####
# n_sentence_length(df, 'My_Cat')
# 




class DataVisualizer2:
    def __init__(self, l, 
                 specific_name = 'test',
                 path_folder = './Data/Locally/', 
                 specific_path = './Data/Locally/AMAZON_FASHION_5.json',
                 myData = None):
        
        self.specific_path = specific_path
        self.specific_name = specific_name

        self.df = data_preprocessing.json_to_df(self.specific_path) 
        self.data = data_preprocessing.dataFormatting(self.df) 
        self.path_folder = path_folder
        self.l = l
        self.myData = myData

    def load_data(self, myData):
        self.myData = myData 

    def load_list(self, l):
        self.l = l
    
    # Plots for list of categories
    def plot_avg_sentences(self):
        cat_avg_sentence(self.l, self.path_folder)

    def plot_review_ratio(self):
        reviewRatio(self.l, self.path_folder)


    # Plots for specific Category
    def plot_common_words(self):
        """
        Analyzes the text data in the 'text' column of the provided DataFrame and
        prints the top 10 most common words along with their frequencies.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

        Returns:
            None

        Execution: 
            Show plot
        """
        common_words(self.myData)

    def n_sentence_length(self):


        n_sentence_length(self.df, 'Category Name')





###################################
##### Old functions saved ######### 
#### In case of safety ############
###################################



def SAVEDreviewRatio(df):
    """
    Analyzed the overall ratings in the 'overall' column of the provided DataFrame,
    and show the counted ratings 1-5.
    """

    value_counts = df['overall'].value_counts().to_dict()
    total_reviews = value_counts[1] + value_counts[2] + value_counts[3] + value_counts[4] + value_counts[5]
    One_procent = 100 * (value_counts[1] / total_reviews)
    Two_procent = 100 * (value_counts[2] / total_reviews)
    Three_procent = 100 * (value_counts[3] / total_reviews)
    Four_procent = 100 * (value_counts[4] / total_reviews)
    Five_procent = 100 * (value_counts[5] / total_reviews)
    # total_procent = One_procent + Two_procent + Three_procent + Four_procent + Five_procent


    # set width of bar 
    barWidth = 0.15
    fig = plt.subplots(figsize =(12, 8)) 
     
    ##### Set values: Absolute values 
    # One = [value_counts[1], value_counts[1], value_counts[1], value_counts[1], value_counts[1]]
    # Two = [value_counts[2], value_counts[2], value_counts[2], value_counts[2], value_counts[2]]
    # Three = [value_counts[3], value_counts[3], value_counts[3], value_counts[3], value_counts[3]]
    # Four = [value_counts[4], value_counts[4], value_counts[4], value_counts[4], value_counts[4]]
    # Five = [value_counts[5], value_counts[5], value_counts[5], value_counts[5], value_counts[5]]

    One = [One_procent, One_procent, One_procent, One_procent, One_procent]
    Two = [Two_procent, Two_procent, Two_procent, Two_procent, Two_procent]
    Three = [Three_procent, Three_procent, Three_procent, Three_procent, Three_procent]
    Four = [Four_procent, Four_procent, Four_procent, Four_procent, Four_procent]
    Five = [Five_procent, Five_procent, Five_procent, Five_procent, Five_procent]




    # Set position of bar on X axis 
    br1 = np.arange(len(One)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 
     
    # Make the plot
    plt.bar(br1, One, color ='r', width = barWidth, 
            edgecolor ='grey', label ='One') 
    plt.bar(br2, Two, color ='g', width = barWidth, 
            edgecolor ='grey', label ='Two') 
    plt.bar(br3, Three, color ='b', width = barWidth, 
            edgecolor ='grey', label ='Three') 
    plt.bar(br4, Four, color ='black', width = barWidth, 
            edgecolor ='grey', label ='Four') 
    plt.bar(br5, Five, color ='purple', width = barWidth, 
             edgecolor ='grey', label ='Five') 
    #  
     
    catgories = ['AMAZON_FASHION', 'ARTS_CRAFT', 'CLOTHING_SHOES', 'MAGAZINE_SUB', 'LUXURY_B']

    # Adding Xticks 
    plt.xlabel('Categories', fontweight ='bold', fontsize = 15) 
    plt.ylabel('N Reviews', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(len(One))], 
            l)
     
    plt.legend()
    plt.show()
    

    return None


