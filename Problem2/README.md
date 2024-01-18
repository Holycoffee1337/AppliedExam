# Problem 2: Text classification
NOTE: get_Data_From_Local_Json() -> Keep adding. Delete files before hand

# Table of Contents
- [Problem 2: Text Classification](#problem-2-text-classification)
  - [Links to Download](#links-to-download)
  - [Data](#data)
  - [Usage](#usage)
  - [Visualize](#visualize)





This project, provide code for text classification.

## Links to download:

https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_jewelry_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz


## Links to 10 smallest:
Gift Cards (147,194 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Gift_Cards_5.json.gz

Magazine Subscriptions (89,689 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz

Software (459,436 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Software_5.json.gz

Prime Pantry (471,614 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Prime_Pantry_5.json.gz

Digital Music (1,584,082 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz

Musical Instruments (1,512,530 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz

Industrial and Scientific (1,758,333 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz

Video Games (2,565,349 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz

Appliances (602,777 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Appliances_5.json.gz

Arts, Crafts and Sewing (2,875,917 reviews):
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz









# Data
Amazon Fashion	reviews (883,636 reviews)	metadata (186,637 products)
All Beauty	reviews (371,345 reviews)	metadata (32,992 products)
Appliances	reviews (602,777 reviews)	metadata (30,459 products)
Arts, Crafts and Sewing	reviews (2,875,917 reviews)	metadata (303,426 products)
Automotive	reviews (7,990,166 reviews)	metadata (932,019 products)
Books	reviews (51,311,621 reviews)	metadata (2,935,525 products)
CDs and Vinyl	reviews (4,543,369 reviews)	metadata (544,442 products)
Cell Phones and Accessories	reviews (10,063,255 reviews)	metadata (590,269 products)
Clothing Shoes and Jewelry	reviews (32,292,099 reviews)	metadata (2,685,059 products)
Digital Music	reviews (1,584,082 reviews)	metadata (465,392 products)
Electronics	reviews (20,994,353 reviews)	metadata (786,868 products)
Gift Cards	reviews (147,194 reviews)	metadata (1,548 products)
Grocery and Gourmet Food	reviews (5,074,160 reviews)	metadata (287,209 products)
Home and Kitchen	reviews (21,928,568 reviews)	metadata (1,301,225 products)
Industrial and Scientific	reviews (1,758,333 reviews)	metadata (167,524 products)
Kindle Store	reviews (5,722,988 reviews)	metadata (493,859 products)
Luxury Beauty	reviews (574,628 reviews)	metadata (12,308 products)
Magazine Subscriptions	reviews (89,689 reviews)	metadata (3,493 products)
Movies and TV	reviews (8,765,568 reviews)	metadata (203,970 products)
Musical Instruments	reviews (1,512,530 reviews)	metadata (120,400 products)
Office Products	reviews (5,581,313 reviews)	metadata (315,644 products)
Patio, Lawn and Garden	reviews (5,236,058 reviews)	metadata (279,697 products)
Pet Supplies	reviews (6,542,483 reviews)	metadata (206,141 products)
Prime Pantry	reviews (471,614 reviews)	metadata (10,815 products)
Software	reviews (459,436 reviews)	metadata (26,815 products)
Sports and Outdoors	reviews (12,980,837 reviews)	metadata (962,876 products)
Tools and Home Improvement	reviews (9,015,203 reviews)	metadata (571,982 products)
Toys and Games	reviews (8,201,231 reviews)	metadata (634,414 products)
Video Games	reviews (2,565,349 reviews)	metadata (84,893 products)



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


## Visualize 
```Python
##### Visiualize Data #####                                                   
DataVisualizer.plot_common_words()                                          
DataVisualizer.plot_avg_n_word_reveiw()                                     
DataHandler.remove_stop_words_in_combined() # Remove stop words in combined 
DataVisualizer.plot_avg_n_word_reveiw()                                     
DataVisualizer.plot_common_words()                                          
DataVisualizer.plot_n_sentence_length()                                     
DataVisualizer.plot_review_ratio()                                          
```
```terminal
Load all categories into class
==============================
Loaded: Magazine_Subscriptions
Loaded: Clothing_Shoes_and_jewelry
Loaded: Combined categories into one dataframe

===========================
Remove stop words from data
===========================

===============================
Saving DataHandler Class
File saved as: DATA_CLASS_SMALL
===============================

=================================================================================================
Data Summary for DATA_CLASS: Saved as csv in path './RESULTS/SUMMARY/data_summary_DATA_CLASS.csv'
=================================================================================================

                  Category  Train Set Size  Validation Set Size  Combined Set Size                                      Train Distribution                         Validation Distribution
    Magazine_Subscriptions            1900                  475               2375 {0: 81, 1: 95, 2: 190, 3: 300, 4: 1234}                 {0: 21, 1: 23, 2: 49, 3: 75, 4: 307}
Clothing_Shoes_and_jewelry          382375                95594             477969 {0: 16427, 1: 17986, 2: 32366, 3: 69479, 4: 246117}     {0: 4116, 1: 4497, 2: 8092, 3: 17357, 4: 61532}
                All_Beauty            4215                 1054               5269 {0: 91, 1: 51, 2: 88, 3: 266, 4: 3719}                  {0: 24, 1: 13, 2: 21, 3: 66, 4: 930}
    Arts_Crafts_and_Sewing          395588                98897             494485 {0: 11491, 1: 10227, 2: 22894, 3: 48918, 4: 302058}     {0: 2867, 1: 2557, 2: 5734, 3: 12227, 4: 75512}
            AMAZON_FASHION            2540                  636               3176 {0: 93, 1: 74, 2: 270, 3: 376, 4: 1727}                 {0: 24, 1: 19, 2: 67, 3: 95, 4: 431}

Test Data Summary (Luxury_Beauty):
 Size                                   Distribution
34278 {1: 1095, 2: 1496, 3: 3884, 4: 7833, 5: 19970}

=================================================================================================

=========================================================================================================
Data Summary for DATA_CLASS_UNI: Saved as csv in path './RESULTS/SUMMARY/data_summary_DATA_CLASS_UNI.csv'
=========================================================================================================

                  Category  Train Set Size  Validation Set Size  Combined Set Size                                     Train Distribution                         Validation Distribution
    Magazine_Subscriptions             405                  405                810 {0: 81, 1: 81, 2: 81, 3: 81, 4: 81}                    {0: 19, 1: 18, 2: 44, 3: 64, 4: 260}
Clothing_Shoes_and_jewelry           82135                82135             164270 {0: 16427, 1: 16427, 2: 16427, 3: 16427, 4: 16427}     {0: 3513, 1: 3858, 2: 6961, 3: 14909, 4: 52894}
                All_Beauty             255                  255                510 {0: 51, 1: 51, 2: 51, 3: 51, 4: 51}                    {0: 5, 1: 4, 2: 8, 3: 14, 4: 224}
    Arts_Crafts_and_Sewing           51135                51135             102270 {0: 10227, 1: 10227, 2: 10227, 3: 10227, 4: 10227}     {0: 1489, 1: 1303, 2: 2928, 3: 6437, 4: 38978}
            AMAZON_FASHION             370                  370                740 {0: 74, 1: 74, 2: 74, 3: 74, 4: 74}                    {0: 13, 1: 12, 2: 45, 3: 58, 4: 242}

Test Data Summary (Luxury_Beauty):
 Size                                   Distribution
34278 {1: 1095, 2: 1496, 3: 3884, 4: 7833, 5: 19970}

=========================================================================================================

=============================================================================================================
Data Summary for DATA_CLASS_SMALL: Saved as csv in path './RESULTS/SUMMARY/data_summary_DATA_CLASS_SMALL.csv'
=============================================================================================================

                  Category  Train Set Size  Validation Set Size  Combined Set Size                                      Train Distribution                         Validation Distribution
    Magazine_Subscriptions            1900                  475               2375 {0: 81, 1: 95, 2: 190, 3: 300, 4: 1234}                 {0: 21, 1: 23, 2: 49, 3: 75, 4: 307}
Clothing_Shoes_and_jewelry          382375                95594             477969 {0: 16427, 1: 17986, 2: 32366, 3: 69479, 4: 246117}     {0: 4116, 1: 4497, 2: 8092, 3: 17357, 4: 61532}

Test Data Summary (Luxury_Beauty):
 Size                                   Distribution
34278 {1: 1095, 2: 1496, 3: 3884, 4: 7833, 5: 19970}

=============================================================================================================:w

```




