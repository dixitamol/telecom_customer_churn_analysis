# **********************************************************************************************
# Load working directory
# **********************************************************************************************
options(warn=-1)
rm(list=ls()) # clears all objects in "global environment"
cat("\014") # clears the console area
#setwd("~//Amol_6562752_V3/code_V1") #FOCUS
getwd() # get working directory

# options(warn=-1) # Can turn the warnings as OFF if needed 

# **********************************************************************************************
# Load libraries
# **********************************************************************************************

Libraries <- c("keras", "tidyquant", "rsample", "recipes", "yardstick", "corrr", "dplyr", 
				 "party", "miscset", "magrittr", "partykit","pROC", "e1071","randomForest",
                 "xlsx","tidyverse","readxl","caret","lime","plyr","gridExtra","ggthemes",
                 "funModeling", "corrplot","ggplot2")	
library(pacman)
pacman::p_load(char=Libraries,install=TRUE,character.only=TRUE)

source("Functions.R")
source("ML_Models.R")
source("Visualisations.R") 

# **********************************************************************************************
# # Data Exploration
# **********************************************************************************************

# Import dataset
df_churn <- read.csv("Telco-Customer-Churn.csv")

print("Data Exploration")
df_churn %>% glimpse()
df_churn %>% df_status()

df_statistics<-calStats(subset(df_churn, select = -c(customerID))) # Calculate statistics
# view(df_statistics)

# **********************************************************************************************
# # Data Pre-processing
# **********************************************************************************************

# Preprocess the data
df_churn_prep <- generalPreprocessingFunction(df_churn)

#Some visualisations
visualiseData <- 1
if (visualiseData == 1){
  print("Data Visualisation is done by plotting the Target field against various features")
  play1(df_churn)      # Distribution of Churn against TotalCharges and log(TotalCharges)
  play2(df_churn)      # Distribution of Churn against MonthlyCharges, tenure and (tenure*MonthlyCharges)
  play3(df_churn_prep) # Distribution of Churn against several categorical features
}
# Create train and test datasets
split <- Split_Function(df_churn_prep,0.7) # Splitting 70:30
train_data <- training(split)
test_data  <- testing(split)

# **********************************************************************************************
# # Run ML models
# **********************************************************************************************

### To run the Keras model put flag = 1 and execute
run_Keras <- 1
if (run_Keras == 1) {

  # Create and bake a recipe for Keras preprocessing data: 
  # convert categorical to dummy, normalise ordinals, ...
  df_baked <- createAndBake_Keras(train_data, test_data) 
  # Run keras model (KerasMetrics - output with model results)
  KerasMetrics <- RunKerasModel(df_baked)
  
  # print calculated measures
  print("Keras calculated measures:")
  print(KerasMetrics$calcMeasures)
  
  # Taking out the predictions and actuals to make the ROC curve
  predicted <- KerasMetrics$ListChurn$predicted
  expected  <- KerasMetrics$ListChurn$expected
  predicted <- ifelse( predicted == "Yes", 1, 0)
  expected  <- ifelse( expected == "Yes", 1, 0)
  
  # Calculating the ROC characteristics : like threshold, AUC
  threshold<- NROCgraph(expected, predicted)
} # End Keras

### To run the Decision Tree model put flag = 1 and execute
run_DT <- 1
if (run_DT == 1) {
  # Results from the Decision Tree model are stored in dtMetrics
  dtMetrics <- RunDecisionTree(train_data, test_data)
    
  # print calculated measures
  print("Keras calculated measures:")
  print(dtMetrics$calcMeasures)
  
  # Taking out the predictions and actuals to make the ROC curve
  predicted <- ifelse( dtMetrics$predicted_vec == "Yes", 1, 0)
  expected <-  ifelse( dtMetrics$expected_vec == "Yes", 1, 0)
  
  # Calculating the ROC characteristics : like threshold, AUC
  threshold<- NROCgraph(expected, predicted)
} # End Decision tree

### To run the Random Forest model put flag = 1 and execute
run_RF <- 1
if (run_RF == 1) {
  # Results from the Random Forest model are stored in rfMetrics
  rfMetrics <- RunRandomForest(train_data, test_data)
  
  # Taking out the predictions and actuals to make the ROC curve
  predicted <- ifelse( rfMetrics$predicted == "Yes", 1, 0)
  expected <- ifelse( rfMetrics$expected == "Yes", 1, 0)
  
  # Calculating the ROC characteristics : like threshold, AUC
  threshold<- NROCgraph(expected, predicted)
} # End random Forest

### To run the Logistic Regression model put flag = 1 and execute
run_LR <- 1
if (run_LR == 1) {
  # Results from the Logistic Regression model are stored in lrMetrics
  lrMetrics <- RunLogisticRegression(train_data, test_data)
  
  # Taking out the predictions and actuals to make the ROC curve
  expected <- ifelse( lrMetrics$expected == "Yes", 1, 0)
  
  # Calculating the ROC characteristics : like threshold, AUC
  threshold<- NROCgraph(expected, predicted)
} #end logistic regression

     ############ END OF MAIN MODELLING FILE############ 
############ ############ THANK YOU ############ ############ 
