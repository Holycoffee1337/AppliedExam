# Todo
- [x] Fixe memory
  - Checke, at dataen faktisk er rigtig
- [x] Fixe saa all grafer virker
- [x] Fixe Class DataLoaderSavedLocally, so it check if already downloaded
- [x] Fix: If category is wrong. DataHandlers list. get updated
- [x] Create terminal print statements
- [x] Cleanse Code (Refactor)
- [x] Adjust Model Class to take datahandler
- [x] def: Filter common words
- [x] Combine og shuffle entire dataset
- [x] Lave test data
- [x] Fixe fejl i data (Det var - fordi det bliver ved at gemme i allerede eksisterende filer)
- [x] def: Split rating
Snak om 
- [x] Implement: Rating split combined 20-20-20-20-20: ? Evt. Kig paa større datasets? 
- [x] Hyperparameter search... ?? not sure how to implement
- [Ikke vitigt] Rewrite: Plot, saa den kan tage flere cat end 5 


# Todo 2
- [x] Create Filter: Ind i -> Combine (20-20-20-20-20 split)
- [] Create a small dataset: That you can test on
- [x] Create DataClas_Yes20 -> With 20-20-20-20-20 split
- [x] Create DataClass_No20 -> Without Rating split 
- [x] Average_words_plot ->> Fjerne stop_words 
- [x] Download all json filer -> Tage en lille mængde data af alle categorier (-> Lave en ny Class data med det)
- [x] Implementer shallow learning models 
  - [x] SVM
  - [x] Boosting
  - [x] Decisions Trees
  - [x] RandomForest 
- [] Implementer test data ind i shallow learning modellerne 
- [] Implementer recall .. og alt det som SVM Giver paa hvert kategori. Saa vi kan se om vi gør modellen mere robust
  - Check: Implementation for shallow with this

# Todo 3 
DataHandler
- [] Update_balance_rating: Fixe split
- [] _Combine_data: Fixe split
      - [] Combine_data: Behøver ikke at have Error handling
      - [] Combined_data: Skal shuffles
- [Ikke vigtigt] Adjuste rating fra: 0-4 -> 1-5?
- [] Implementere -> Set all files to nothing or delete
- [Ikke vigtigt] Maaske gøre saa den slet ikke gemmer train- val filerne? 

Shallow-Learning 
- [] Do so they can evaulate on test data
- [] Check If pipeline is faster








## Todo: Maybe
- [] Create a stats table: To show: ratings before -> and after filter. How small the datasets gets  

# Huske noter: 
- !!!! Muligvis fejl i at jeg generere X_train, Y_train, X_val, Y_val -> ?? i den maade jeg haandtere det paa 
  - Fejl i hvor jeg gemmer de fire filer. og Concatenater efter DUMT DUMT DUMT
    - For at haandtere split i selv modellen.


# Overall Ide
- 2 Data Class  
  # 20-20-20-20-20: Split og uden
  - 1) Normal: 5 Categorier: -> Dem der minder om test. uden vider behandling  + stop_words removal?
  - 3?) Alle Categorier: 20-20-20-20-20: Alle categorier fylder lige meget?
- Teste begge ting


# Shallow-Learning models
- Base modellerne klarer sig nok bedre. Paa data hvor der ikke er 20-20-20-20-20 split
  - Nemmere ved at overfitte. Højt Sandsynligt. Klare RNN modellen sig bedre paa det komplicret data
- def: split rating 20-20-20-20-20

## Noter til rapport
- husk og nævn summaray + ... er = Text



## Code Structure notes
- Class in top
- remember to delete unnec
