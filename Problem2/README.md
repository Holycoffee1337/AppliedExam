# Problem 2: Text classification
NOTE: get_Data_From_Local_Json() -> Keep adding. Delete files before hand


This project, provide code for text classification.

## Links to download:

https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_jewelry_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz


## Usage
```Python
# Exuecute if its first time creating the data
data_preprocessing.DataloaderSavedLocally(CATEGORIES)

##### Creating our 2 classes #####
DataHandler = data_preprocessing.DataHandler(CATEGORIES)
DataHandler.get_Data_From_Local_Json()
DataVisualizer = visualization.DataVisualizer(DataHandler)


##### Load data for entire list #####
DataHandler.load_all_categories()
DataHandler.saveDataHandlerClass('Final_Categories')
     
     
##### Get train-, val- and df data #####
All_Beauty_train_data = DataHandler.get_data(ALL_BEAUTY, 'train')
All_Beauty_val_data = DataHandler.get_data(ALL_BEAUTY, 'val')
All_Beauty_df = DataHandler.get_data(ALL_BEAUTY, 'df') 
   
###### Plot single cateogry data #####
DataVisualizer.plot_common_words(ALL_BEAUTY)
DataVisualizer.plot_n_sentence_length(AMAZON)

##### Plot for entire category #####
DataVisualizer.plot_review_ratio()
DataVisualizer.plot_avg_n_word_reveiw()

```



