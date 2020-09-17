# **********************************************************************************************
# RUNDECISIONTREE()
# Run & test the decision tree model
#
# INPUT:   Dataframes - preprocessed train data, preprocessed test data
# OUTPUT : List - list with confusion matrix & model predictions (accuracy,TPR,FPR,MCC..)
#
# V1.0 Amol Dixit 10/06/2019
# **********************************************************************************************

RunDecisionTree <- function (train_data,test_data ){
  
  # ######## DECISION TREE
  # Generating the Decision Tree using a select group of features only
  tree <- ctree(Churn~Contract+tenure+PaperlessBilling+InternetService, train_data)
  pred_tree <- predict(tree, test_data) # Calclating the prediction metrics on test data
  confMat <- confusionMatrix(pred_tree,test_data$Churn) # Returns the ConfMat in integer form
  # Extracting Rules from the DT model 
  partykit:::.list.rules.party(tree)
  
  matrix <- tab_int2double(confMat$table) # Updating to Double form for further calculations
  # Using the function from the lab to calc performance measures
  measures <- NcalcMeasures(matrix[2,2],matrix[1,2],
                            matrix[2,1],matrix[1,1])
  
  return(list( 'tree'=tree, 'plotTree'=plot(tree), 
               'confMat'=confMat, 'calcMeasures'=measures, 'predicted_vec'=pred_tree,
               'expected_vec'=test_data$Churn))
  
} # END DECISION TREE

# **********************************************************************************************
# RUNRANDOMFOREST()
# Run & test the random forest model
#
# INPUT:   Dataframes - preprocessed train data, preprocessed test data
# OUTPUT : List - list with confusion matrix & model predictions (accuracy,TPR,FPR,MCC..)
#
# V1.0 Amol Dixit 10/06/2019
# **********************************************************************************************

RunRandomForest <- function (train_data,test_data ){
  # Generating the Random Forest
  rfModel <- randomForest(Churn ~., data = train_data)
  pred_rf <- predict(rfModel, test_data) # Calclating the prediction metrics on test data
  confMat <- confusionMatrix(pred_rf,test_data$Churn) # Returns the ConfMat in integer form
  matrix <- tab_int2double(confMat$table) # Updating to Double form for further calculations
  # Using the function from the lab to calc performance measures
  measures <- NcalcMeasures(matrix[2,2],matrix[1,2],
                            matrix[2,1],matrix[1,1])
  return(list('rfModel'=rfModel, 'plotrfModel'=plot(rfModel),
              'confMat'=confMat, 'calcMeasures'=measures,
              'predicted'=pred_rf, 'expected'=test_data$Churn))
  
} # END RANDOM FOREST

# **********************************************************************************************
# RUNLOGISTICREGRESSION()
# Run & test the logistic regression model
#
# INPUT:   Dataframes - preprocessed train data, preprocessed test data
# OUTPUT : List - list with confusion matrix & model predictions (accuracy,TPR,FPR,MCC..)
#
# V1.0 Amol Dixit 10/06/2019
# **********************************************************************************************

RunLogisticRegression <- function (train_data,test_data ){
  cross_validation = 0
  test_data_reg <- test_data
  train_data_reg <- train_data
  Churn_train <- ifelse(pull(train_data_reg, Churn) == "Yes", 1, 0)
  Churn_test  <- ifelse(pull(test_data_reg, Churn) == "Yes", 1, 0)
  
  if (cross_validation==0){
    # Creating Model
    LReg_Model <- glm(Churn ~ .,family=binomial(link="logit"),data=train_data_reg)
    
  }else{  
    modelCtrl <- trainControl(method = "cv", number = 3)
    #Logistic
    set.seed(123)
    # Creating Model
    LReg_Model <- train(Churn ~ ., method = "glm", family = "binomial", 
                        data = train_data_reg, trControl = modelCtrl)
    summary(LReg_Model)
    varImp(LReg_Model)
  }
  
  # Printing the model
  print(summary(LReg_Model))
  
  # Creating Anova
  anova(LReg_Model, test="Chisq")
  
  # Generating Log Odds Ratio
  exp(cbind(OR=coef(LReg_Model), confint(LReg_Model)))
  
  # Making Predictions
  fitted_results <- predict(LReg_Model,newdata=test_data_reg,type='response')
  
  # Formatting to 0s and 1s
  fitted_results <- ifelse(fitted_results > 0.5,1,0)
  
  # Confusion Matrix
  confMat<- table(expected=test_data_reg$Churn, predicted=fitted_results > 0.5)
  
  # TRY USING THE FUNCTION CONVERT THE VALUES IN DOUBLE
  measures <- NcalcMeasures(TP=as.double(confMat[2,2]),FP=as.double(confMat[1,2]),
                FN=as.double(confMat[2,1]),TN=as.double(confMat[1,1]))
  
  return(list('confMat'=confMat, 'calcMeasures'=measures, 
              'expected'=test_data_reg$Churn, 'predicted'=fitted_results))
  
} # END LOGISTIC REGRESSION

# **********************************************************************************************
# RUNKERASMODEL() :
# Run the deep neural network Keras model
#
# INPUT:   Dataframes - df_baked (DF with baked data), 
# OUTPUT : List - list with confusion matrix & model predictions (accuracy,TPR,FPR,MCC..)
#
# V1.0 Amol Dixit 10/06/2019
# **********************************************************************************************

RunKerasModel<-function(df_baked){
  
  # Splitting the baked test and train data
  train_data <- subset(df_baked$train_data, select = -c(Churn)) 
  test_data  <- subset(df_baked$test_data, select = -c(Churn))
  # Separating the target vector
  tgt_train <- ifelse(pull(df_baked$train_data, Churn) == "Yes", 1, 0)
  tgt_test  <- ifelse(pull(df_baked$test_data, Churn) == "Yes", 1, 0)

  # Initialising the Keras Model
  model_keras <- keras_model_sequential()
  
  # Running Keras Model with 2 hidden layers
  model_keras <-  model_keras %>% 
    
    # First hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu", 
      input_shape        = ncol(train_data)) %>% 
    
    # Dropout to prevent overfitting
    layer_dropout(rate = 0.1) %>%
    
    # Second hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu") %>% 
    
    # Dropout to prevent overfitting
    layer_dropout(rate = 0.1) %>%
    
    # Output layer
    layer_dense(
      units              = 1, 
      kernel_initializer = "uniform", 
      activation         = "sigmoid") %>% 
    
    # Compile ANN
    compile(
      optimizer = 'adam',
      loss      = 'binary_crossentropy',
      metrics   = c('accuracy')
    )
  
  # Printing the model
  model_keras
  
  # Generating the history by using fit() function
  history <- keras::fit(
    object           = model_keras, 
    x                = as.matrix(train_data), 
    y                = tgt_train,
    batch_size       = 50, 
    epochs           = 20,
    validation_split = 0.30
  )
  
  # Printing the history
  print(history)
  
  # Plotting the model
  plot(history)
  
  # -------------------
  # Making Predictions 
  # -------------------                                              

  # Creating Predictions Vectors
  keras_class <- predict_classes(object = model_keras, x = as.matrix(test_data)) %>%
    as.vector()
  keras_prob  <- predict_proba(object = model_keras, x = as.matrix(test_data)) %>%
    as.vector()
  
  # Creating predicteds
  ListChurn <- tibble(
    expected      = as.factor(tgt_test) %>% fct_recode(Yes = "1", No = "0"),
    predicted   = as.factor(keras_class) %>% fct_recode(Yes = "1", No = "0"),
    class_prob = keras_prob
  )
  
  # Creating Confusion Matrix
  confmat <- ListChurn %>% conf_mat(expected, predicted)
  cm <- confmat$table
  # Using the function from the lab to calc performance measures
  measures <- NcalcMeasures(as.double(cm[2,2]),cm[1,2],
                            cm[2,1],cm[1,1])

  return(list( 'ListChurn'=ListChurn,'conf_mat'=confmat$table, 'calcMeasures' = measures))
  
} #ENDOF RUNKERASMODEL()

       ############ END OF MODELS FILE############ 
############ ############ THANK YOU ############ ############ 