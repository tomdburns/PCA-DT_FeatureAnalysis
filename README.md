# PCA-DT_FeatureAnalysis
Performs PCA and DT fittings to assess feature importance

 ===========================================
 
 README File for PCA.py and DecisionTree.py codes
 
 Code Author: Tom Burns
 
 email: tom.burns@canada.ca
 
 Date: Feb-24th 2021
 
 VERSION 1.1.0
 
 ===========================================
 
 ### Table of Contents
 
 1. Installing Python and required Modules
 2. Setting up Codes
 3. Running the Codes

 ===========================================
 
 ### 1. Installing Python and Required Modules
 -------------------------------------------
 #### Step 1: Installing Python
 - Download and Install python3.9 or earlier installer from: https://www.python.org/
   
 #### Step 2: Installing Modules
 - to install a module, from the command line type
 
    pip --install MODULE
    
    where MODULE is the name of the module you are trying to install
    
 - List of required modules:
   1) numpy
   2) pandas
   3) matplotlib
   4) sklearn
 
 ==========================================
 
 ### 2. Setting Up Codes
 ------------------------------------------
 To run the codes, simply edit the top variables in both codes:
 
 *INFILE* - defines the name of the input file that contains your descriptors and
          target values. THIS FILE NEEDS TO BE IN CSV FORMAT (comma delimited)
	  
 *TARGET* - The name of the column in your csv file that corresponds to the target
          classifications
	  
 *OMIT*   - A list of columns in your CSV file you do not want considered in the
          analysis (example: Substance Name)
	  
 *SCALE*  - Do you want the descriptors to be scaled using the Standard Scaler?
          (set to True or False, recommended: True)
 *NTREES* - The number of decision trees you would like to run (only in DecisionTree.py)
 
          - if NTREES > 1, a random selection of descriptors will be selected for
		       each tree.
		       
		    - if NTREES = 1, all descriptors will be considered in fitting the tree
		    
          - Default value is 10,000 Trees
	  
 *BOOT*   - Run a boostrap to calculate the 95% CI for the decision tree accuracies
          (Only in DecisionTree.py
          - Default = False
	  
 *BSTEPS* - Controls the number of boostrapping steps performed in 95% CI calculation
 
          - Default = 1000
	 
 ==========================================
 
 ### 3. Running the Codes
 ------------------------------------------
 #### From a directory:
 
 1. Open the folder containing the python code
 
 2. make sure the csv containing your data is located in this folder
 
 3. change the INFILE variable in PCA.py and DecisionTree.py to match the
    name of your csv file containing the data and target values

 4. Double click PCA.py or DecisionTree.py to run the code
 
 
 #### From a command line:
 
 1. To run this code, make sure your csv file containing the data is located in
 the same directory as the code.
 
 2. change the INFILE in PCA.py and DecisionTree.py to match the name of your
    csv file containing the data and target values
 
 3. in a Terminal window, navigate to the directory containing the code
	- if on Windows, I recommend PowerShell

 4. Once in the correct directory, type:
 
		python PCA.py 			or			python DecisionTree.py
 ==========================================
