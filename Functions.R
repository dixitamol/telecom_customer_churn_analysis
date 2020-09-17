
# ********************************************************************************************************
# generalPreprocessingFunction() 
#   
# Initial data pre-processing for the Churn dataframe, steps:
#     1 - Delete field customerID as it is unique
#     2 - Removing rows with tenure less than 1 month to deal with missing values
#     3 - Field creation: create new field TotalChargesUpdated (combination of TotalCharges 
#                         MonthlyCharges and tenure as this are very highly correlated) 
#     4 - Remove fields TotalCharges and MonthlyCharges 									
#     5 - Field discretisation: Field tenure measured in months will be converted to years	
#     	 					    to get better data mining results
#     6 - SeniorCitizen values {0,1} converted to {"Yes","No"} to match the other categorical fields format
#     7 - Update values  
#        Value "No internet service" redundant for: OnlineSecurity, OnlineBackup, 
#                 DeviceProtection, TechSupport, streamingTV, streamingMovies
#        NOTE: All these columns are dependent on InternetService=Yes 
#        Value "No phone service" redundant for column MultipleLines
#   
# INPUT:   dataframe - dataset
# OUTPUT:  dataframe - Pre-processed data frame
#
# V1.0 Amol Dixit 10/06/2019
# ********************************************************************************************************

generalPreprocessingFunction <- function(df_churn){
    
  df_churn <- subset(df_churn, select = -c(customerID)) 
  df_churn <- subset(df_churn,tenure>0) # Removing rows with tenure less than 1 month to deal with missing values
  
  check_corr = 0
  if (check_corr == 1){
  # Check correlation between TotalCharges~MonthlyCharges 
    df_churn %>%
      dplyr::select(TotalCharges, tenure,MonthlyCharges) %>%
      mutate(
        TotalMonths = tenure*MonthlyCharges,
        TotalCharges = log(TotalCharges)
      ) %>%
      corrr::correlate() %>%
      corrr::focus(TotalMonths) %>%
      fashion() }
  
  print("As TotalMonths and TotalCharges are highly correlated (0.83) combined them to TotalChargesUpdated")
  # creating a new field with the mean value of both
  df_churn <-  df_churn %>% mutate(TotalChargesUpdated = (TotalCharges + tenure*MonthlyCharges)/2)
  df_churn <- subset(df_churn, select = -c(MonthlyCharges,TotalCharges))
  
  # ------------------------------
  # discretisation of field tenure
  # ------------------------------ 
  group_tenure <- function(tenure){
    case_when(
      (tenure > 0 & tenure <= 12) ~  1L,
      (tenure > 12 & tenure <= 24) ~ 2L,
      (tenure > 24 & tenure <= 48) ~ 3L,
      (tenure > 48 & tenure <= 60) ~ 4L,
      (tenure > 60) ~ 5L,
      TRUE ~ 0L)}
  
  df_churn$tenure <- sapply(df_churn$tenure, group_tenure)
  df_churn$tenure <- as.numeric(df_churn$tenure)
  
  #Ordinal discretisation for field SeniorCitizen
  df_churn$SeniorCitizen <- as.factor(plyr::mapvalues(df_churn$SeniorCitizen,
                                                from = c("0", "1"), to = c("No", "Yes")))

  # -------------------------------------------------------------------------
  # Update values  
  # Value "No internet service" redundant for: OnlineSecurity, OnlineBackup, 
  #          DeviceProtection, TechSupport, streamingTV, streamingMovies
  # NOTE: All these columns are dependent on InternetService=Yes
  # -------------------------------------------------------------------------
  
    cols <- which(names(df_churn)%in%c("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                    "TechSupport" ,"StreamingTV", "StreamingMovies"))
    
    for (i in 1:ncol(df_churn[, cols])) {
      df_churn[, cols][, i] <- as.factor(mapvalues(df_churn[, cols][, i], from = c("No internet service"), to = c("No")))
    }
    
    # Value "No phone service" redundant for column MultipleLines
    df_churn$MultipleLines <- as.factor(mapvalues(df_churn$MultipleLines, from = c("No phone service"), to = c("No")))
    
    if (any (is.na(df_churn))) df_churn <- drop_na(df_churn) # delete NA if any
    
    return(df_churn)
} # END generalPreprocessingFunction

# ********************************************************************************************************
# calStats() 
#
# Calculate several statistical parameters (standard deviation, percentages..) 
# for the categorical fields of the Churn dataframe 
#
# INPUT:   dataframe - dataset
# OUTPUT:  dataframe - dataset statistics for categorical fields   
#
# V1.0 Amol Dixit 10/06/2019
# ********************************************************************************************************

calStats<-function(df){
    
  # Deleting the numeric features  
  numfields <- select_if(df, is.numeric) %>%  names()
  cols <- which(names(df)%in%numfields)
  df[, cols]<- NULL
  
  variables <- colnames(df)
	# create data frame for results
	df_Statistics <- as.data.frame(c(1:7)) 
	df_Statistics <- t(df_Statistics)
	colnames(df_Statistics) <- c("Level", "Variable" ,"Number", "Churn percentage",
					"Standard Deviation", "Number (No Churn)", "Number (Churn)") 
	df_Statistics <- df_Statistics[-1,]
	df$Churn_num <- ifelse(df$Churn=='No',0,1)
	
	# Calculating the actual stats
	m0 <- df[df[, "Churn"] == "No",]
	m1 <- df[df[, "Churn"] == "Yes",]
	
	for (n in 1:ncol(df)) {
  	DFresults <- data.frame(
  							variable=variables[n],
  							level=levels(factor(df[,n])), 
  							number=tapply(df$Churn_num, df[,n], length),
  							Churn_percentage=round(tapply(df$Churn_num, df[,n], mean)*100,2), 
  							Standard_Deviation=tapply(df$Churn_num, df[,n], sd),
  							Num_NoChurn=tapply(m0$Churn_num, m0[,n], length),
  							Num_Churn=tapply(m1$Churn_num, m1[,n], length)
  							)
  	df_Statistics<-  rbind(df_Statistics,DFresults)
	
	}
	return(df_Statistics)
} # END  calStats

# ********************************************************************************************************
#  createAndBake_Keras()
#
#  Creating a recipe for further pre-processing the Churn dataset
#  this recipe will be used by the ML Keras algorithm
#  taken pre-processing steps to run Keras: 
#     - Convert {"Yes","No"} categorical fields to dummy variables
#	  - Center, scale and normalize features
#
# INPUT:   dataframe - Churn pre-processed train & test datasets
# OUTPUT:  dataframe - pre-processed churn dataframe redy to be run by Keras algorithm
#
# V1.0 Amol Dixit 10/06/2019
# ********************************************************************************************************

createAndBake_Keras <- function(train_data, test_data){
    
  # Creating a recipe
  recipe_object <- recipe(Churn ~ ., data = train_data) %>%
    step_log(TotalChargesUpdated) %>%
    step_dummy(all_nominal(), -all_outcomes()) %>%
    step_center(all_predictors(), -all_outcomes()) %>%
    step_scale(all_predictors(), -all_outcomes()) %>%
    prep(data = train_data)
  
  # Baking the recipe
  train_data_x <- bake(recipe_object, new_data = train_data) 
  test_data_x  <- bake(recipe_object, new_data = test_data) 
  
  retList<-list("train_data"=train_data_x,
                "test_data"=test_data_x)
  return(retList)
}

# ********************************************************************************************************
# tab_int2double()
# 
# Function to convert an integer table to double 
#
# INPUT:   table - table of integer numbers
# OUTPUT:  table - table of double numbers
#
# V1.0 Amol Dixit 10/06/2019
# ********************************************************************************************************

tab_int2double <- function(intTab){
  doubleTab <- intTab
  for(i in nrow(intTab)){
    for (j in ncol(intTab)) {
      doubleTab[i,j] <- as.double(intTab[i,j])
    }
  }
  return(doubleTab)
} # END tab_int2double()

# **********************************************************************************************
# SPLIT_FUNCTION()  
# Data sampling into train and test
#
# INPUT: Frame - dataset
# OUTPUT : Train and Test data frame
# **********************************************************************************************
Split_Function <- function(df,proportion){
  set.seed(427) # Setting uncommon seed
  split <- initial_split(df, proportion)
} # END SPLIT_FUNCTION

# ********************************************************************************************************
# NcalcMeasures() 
#
#  Evaluation measures for a confusion matrix
#
# INPUT: numeric TP, FN, FP, TN
# OUTPUT: A list with the following entries:
#        TP - int - True Positive records
#        FP - int - False Positive records
#        TN - int - True Negative records
#        FN - int - False Negative records
#        accuracy - float - accuracy measure
#        pgood - float - precision for "good" (values are 1) measure
#        pbad - float - precision for "bad" (values are 1) measure
#        FPR - float - FPR measure
#        TPR - float - FPR measure
#        MCC - float - Matthew's Correlation Coeficient
#
# V1 Nick Ryman
# ********************************************************************************************************

NcalcMeasures<-function(TP,FN,FP,TN){
  
  NcalcAccuracy<-function(TP,FP,TN,FN){return(100.0*((TP+TN)/(TP+FP+FN+TN)))}
  NcalcPgood<-function(TP,FP,TN,FN){return(100.0*(TP/(TP+FP)))}
  NcalcPbad<-function(TP,FP,TN,FN){return(100.0*(TN/(FN+TN)))}
  NcalcFPR<-function(TP,FP,TN,FN){return(100.0*(FP/(FP+TN)))}
  NcalcTPR<-function(TP,FP,TN,FN){return(100.0*(TP/(TP+FN)))}
  NcalcMCC<-function(TP,FP,TN,FN){return( ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))}
  
  retList<-list(  "TP"=TP,
                  "FN"=FN,
                  "TN"=TN,
                  "FP"=FP,
                  "accuracy"=NcalcAccuracy(TP,FP,TN,FN),
                  "pgood"=NcalcPgood(TP,FP,TN,FN),
                  "pbad"=NcalcPbad(TP,FP,TN,FN),
                  "FPR"=NcalcFPR(TP,FP,TN,FN),
                  "TPR"=NcalcTPR(TP,FP,TN,FN),
                  "MCC"=NcalcMCC(TP,FP,TN,FN)
  )
  return(retList)
} # END NcalcMeasures()

# ********************************************************************************************************
# NROCgraph() 
#
#  This is a ROC graph
#
# INPUT:        Frame - dataset to create model
#               Fame - dataset to test model
# OUTPUT :      Float - calculated thresholkd from ROC
#
# V1 Nick Ryman
# ********************************************************************************************************

NROCgraph<-function(expected,predicted){
  
  roc_title = "ROC Chart for keras model"
  rr<-roc(expected,predicted,
          plot=TRUE,auc=TRUE, auc.polygon=TRUE,
          percent=TRUE, grid=TRUE,print.auc=TRUE,
          main=roc_title)
  plot(rr)
  #Selects the "best" threshold for lowest FPR and highest TPR
  analysis<-coords(rr, x="best",best.method="closest.topleft",
                   ret=c("threshold", "specificity",
                         "sensitivity","accuracy",
                         "tn", "tp", "fn", "fp",
                         "npv","ppv"))
  
  fpr<-round(100.0-analysis["specificity"],digits=2L)
  threshold<-analysis["threshold"]
  
  #Add crosshairs to the graph
  abline(h=analysis["sensitivity"],col="red",lty=3,lwd=2)
  abline(v=analysis["specificity"],col="red",lty=3,lwd=2)
  
  #Annote with text
  text(x=analysis["specificity"],y=analysis["sensitivity"], adj = c(-0.2,2),cex=1,
       col="red",
       paste("Threshold: ",round(threshold,digits=4L),
             " TPR: ",round(analysis["sensitivity"],digits=2L),
             "% FPR: ",fpr,"%",sep=""))
  
  return(threshold)
}

       ############ END OF FUNCTIONS FILE############ 
############ ############ THANK YOU ############ ############ 