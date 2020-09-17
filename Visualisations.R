options(warn=-1)

play1 <- function(df_churn){
  
  # get the numerical features
  
  num_features <- select_if(df_churn, is.numeric) %>%  names()
  
  ### Distribution of TotalCharges and log(TotalCharges) in relation to Churn
  df_churn %>% 
    dplyr::select(Churn, TotalCharges) %>%
    mutate(
      Log_TotalCharges = log(TotalCharges)) %>%
    gather(key = key, value = value,-Churn) %>% 
    ggplot(aes(x = value,  fill = Churn))+
    geom_histogram(alpha = 0.9)+
    facet_wrap(~ key, scales = 'free')+
    theme_stata()+
    scale_fill_tableau()+
    theme(axis.text.y = element_text(angle = 360))+
    labs(title = 'Distribution of numerical features in relation to Churn',
         x = '',y='')
  
}

play2 <- function(df_churn){
  num_features <- select_if(df_churn, is.numeric) %>%  names()
  
  ### Distribution of tenure, MonthlyCharges, TotalMonthlyCharges in relation to Churn 
  df_churn %>% 
    dplyr::select(Churn, c(tenure, MonthlyCharges)) %>%
    mutate(TotalMonthlyCharges = tenure * MonthlyCharges) %>%
    gather(key = key, value = value,-Churn) %>% 
    ggplot(aes(x = value,  fill = Churn))+
    geom_histogram(alpha = 0.9)+
    facet_wrap(~ key, scales = 'free')+
    theme_stata()+
    scale_fill_tableau()+
    theme(axis.text.y = element_text(angle = 360))+
    labs(title = 'Distribution of numerical features in relation to Churn',
         x = '',y='')
}



play3 <- function(df_churn){
  categorical_features <- df_churn %>%   select_if(is.factor) %>%   names()
  
  df_churn %>% 
    dplyr::select(one_of(categorical_features)) %>% 
    gather(key = key, value = value, - Churn, factor_key = T) %>% 
    ggplot(aes( x = value, fill = Churn))+
    geom_bar()+ 
    facet_wrap(~key, scales = 'free')+
    theme_stata()+
    scale_fill_tableau()+
    scale_x_discrete(labels = abbreviate)+
    theme(axis.text.y = element_text(angle = 360))+
    labs(title = 'Distribution of categorical features in relation to Churn',
         x = '',y='')
}


    ############ END OF VISUALISATIONS FILE############ 
############ ############ THANK YOU ############ ############ 