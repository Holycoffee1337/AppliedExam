# Noter til rapporten
Nævne:
- Chunk size, for bedre haandtering af data 
- Saving DataHandler Class (Once made) -> We can save multiple. To test different categories
      - Among other things
      - pickle -> Optimize: Faster than readon JSON text formats
- Kombine summary og text review til en
- remove stopwords
- Robust -> 20-20-20-20-20


# Noter til selv
- Fejl: df_read_save_json -> Only works with a big chunk -> Error: When get few in some categories. (only 1)


- Average Number of Words for different Categories
    - Medregner ikke: Remove of stop words
- Vi skal nok ind og have de større filer


















# Udvælgelse
- Vi udvælger kategorier der minder om test:
  - Amazon_Fashion
  - All_beauty
  - Clothing_Shoes_and_jewelry
  - Magazine_clothing


- Review Length Analyis (For each category)
  - Output: Output for average sentences length - for each category 
  - Output: Graph over length over all sentence. So we know, how much we should take.





Log -> Til udvælgelse af sampling fra categorier: Saa store kategorier er mere repræsenteret 



# Noter
## Overall 
Evaluation:
  - On the decisions
  - Maybe use scientific articles? 
    - Efficient Net Paper
      - Bigger is not always better
  - He dont want us to be negative about the hardware
    - Make the positive take
  - En opgave ligger op til a bruge pre-trainede models


 - Optimizer: Weight and bias
    - Hyperparameter: Apply Early Stopping
    - Always save - Parameters
    - Approach:
      - Smaller dataset -> ...
      - Big dataset

    - Approach
      - Generel -> To specific
      - Specific -> To Generel

- Code: 
  - Easy to read     
  - CONSTANTS -> TOP


- Submit: 
  - Mention both have done everything equal 
  - Uploade code on Github

  


# Problem 1

Possble ideas
- tf.data.Datasets (Use - ?) (Essentielt)
- SpatialDropout
  - Maybe only use before last layer
    - Have most effect there
- Dropout - increase after each layer
- Skip Connection
- GrayScaling? Make compute faster


- Compare: Normal Resizing with Compressed
  - (He will be dissapointet if auto-encoders dont beat it)





# Problem 2
Data: 2 kinds of sets 
  - Experimental
  - The one you need approval to



 - Shallow: Ensemble
 - Adjust labels: df_train['overall'] - 1
 - Remove Unnecssary words
 - (Think about synonyms)
 - ReduceLRnPlateau( )        - Might speed process up
 - Gøre saa ord, der næsten er det samme. 


Encoder: 
  - Feel free to use another
Word2Vec

ProductCategories: 
  - We can take as many we want


Antagelsen: VI afgrænser modellen, og gaar efter det bedste resultat paa fashion





# Todo
- Adjust labels                       [+]
- Optimizer
- Word2Vec
- Encoder -> Look into that


- Shallow Learning model opsætning




# Discord Samtale
- Data split? 20-20-20-20-20
  - Plot længden af gennemsnittet for de forskellige reviews.

- Refactorisering af koden
- Sætte - Alle base modellerne for shallow learning




# TODO 
- def: Der splitter data op. Saa man har 20-20-20-20-20
- Datahandler -> Ind i datavisualization og model class
- Fixe: visualization for list? af en eller anden grund
  - Og gør funktionerne: Saa navnet paa kategorien kommer paa alle 
- Fixe: wandB
- Downloade alle: Lave en class med alt data der skal bruges og gemme den
- Refaktorisering af kode
- Lave Hyperparameter search

## TODO: Imorgen
# Imogen skal alt kode være "clean"
# Saa rapporten og modeller skal overvejes


  ## Christopher
  - Refaktorisering af kode
  - Slag plan for antagelse: For begge spørgsmaal
  - Tænke over ting vi kunne implementere
  - Implementere Class for alt data ---> Gem den: Det den maade vi haandtere det hele paa (Naar fast besluttet)


  - Sætte mig: Ind i spørgsmaal 1: Og kode (hvis brug for)
  - Tilføje Wandb
  - Tilføje Hyperparameter Search 

  ## Marcell
  - Klare TODO



  ## Begge: 
  - Slag plan for kode "Skrive maade"
  - Læse hinandens kode igennem
  - Sætte GitHub op for koden 
  - Kommenter koden
  - Lave ReadMe.md -> (ogsaa saa vi selv kan huske det)


  - Rens unødvendige filer
  - Ordentlige fil navne
  - ORdentlige variable navne
  - Enig om kommentar struktur
  - Enig om struktur i koden
  - Slette unødvendige imports


  ## Hvis tid
  - Sætte shallow learning modellerne op
    - Tjek alle de modeller han nævner i forlæsningerne



