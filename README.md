#COMP472 A1 README File

Our project contains 8 files. 6 of which are the classes for the ML models. 
The other two files are the main and outFile. 

The outFile is used to create the csv?s with all the information requested in question 3. Namely, it prints the index and predictions, plots the confusion matrix, prints the precision, recall, and f1-measure for each class and  the accuracy, macro-average f1 and weighted-average f1 of the models.

In the main we do several tasks:
-extract all the data from either dataset 1 or 2: line 28 to 65
-plot the distribution for that dataset: line 67 to 71 (question 1)
-call each ML models with the data and call outFile to output the results: line 73 to 89

To run our program, all that is required is to run the main. 
To choose between dataset 1 or 2 all that is needed is to change the dataset_index variable on line 26. Input 1 for DS1 and 2 for DS2. 

## Specifications for certain models:

**bestDT.py** : The class generates the best tree for the 1st dataset by default. In order to perform grid search, uncomment the lines 9 through 19. Uncommenting lines 27 through 34 will train the best model for the 2nd dataset instead.

**bestMLP.py** : This function takes in a set of training and validation data and outputs the resulting preditions for X_test. The function is split into different parts. model() is used to create the classifier with different set of parameters. predictions() function takes in a model and a validation or test data to output preditions from that specific dataset. stats() gets me stats for my parameter comparison. I use bestParms() function to perform a grid search to find the best set of parameters for a given dataset. These parameters are put in a dictionary and are looped and tested, the set of parameters that get me the best results is returned. Since I already performed tests for the best params I commented it out and use specific parameters for given datasets (sets of parameters are specified). Starting from line 51 I put instructions on what to comment and uncomment depending on the dataset.  Parameter output test in presentation. 