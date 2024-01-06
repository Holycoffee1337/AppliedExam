# Main ex
import data_preprocessing
import visualization

# TODO: Chunksize in datahandler, when loading train.. val and so on

CATEGORIES = ['Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # Er ikke downloaded endnu 
            'Arts_Crafts_and_Sewing', 
            'AMAZON_FASHION',
            'Luxury_Beauty']

if __name__ == "__main__":
    AMAZON  = 'AMAZON_FASHION' 
    LUXURY = 'Luxury_Beauty'
    CLOTHING = 'Clothing_Shoes_and_jewelry'
    
    # Load data
    # data_preprocessing.DataloaderSavedLocally()
    ##### Creating our 2 classes #####
    DataHandler = data_preprocessing.DataHandler(CATEGORIES)
    DataVisualizer = visualization.DataVisualizer(DataHandler)

    ##### Load data for a specific category #####
    DataHandler.load_data_for_category(AMAZON)

    ##### Load data for entire list #####
    DataHandler.load_all_categories()
    DataHandler.saveDataHandlerClass('Final_Categories')
    
    
    # Get train-, val- and df data 
    amazon_train_data = DataHandler.get_data(AMAZON, 'train')
    amazon_val_data = DataHandler.get_data(AMAZON, 'val')
    amazon_df = DataHandler.get_data(AMAZON, 'df') 
    current_category = DataHandler.get_current_cateogry()

     # Get train-, val- and df data 
    clothing_train_data = DataHandler.get_data(AMAZON, 'train')
    clothing_val_data = DataHandler.get_data(AMAZON, 'val')
    clothing_df = DataHandler.get_data(AMAZON, 'df') 
    print(clothing_df)
    current_category = DataHandler.get_current_cateogry()

    
    ###### Plot single cateogry data #####
    # DataVisualizer.plot_common_words(AMAZON)
    # DataVisualizer.plot_n_sentence_length(AMAZON)
    DataVisualizer.plot_review_ratio()

    # Plot list
    # DataVisualizer.load_list(CATEGORIES)


    # DataVisualizer.plot_avg_sentences()

#############################
#### Fix plot functions #####
#############################
    

    quit()



