library("caret")
library("readr")
library("ggplot2")
library("GGally")
library("glmnet")
library("e1071")
#1 Load the entire dataset into R and shuffle the dataset appropriately. 
#Report the summaries and comment on any relationships you find among the predictors (X1 - X4) and 
#between each predictor and the outcome. You are welcome to re-use the responses you produced for exam 1.
data<-read_csv("dataset.csv")[-1]
summary(data)
ggpairsdata<-ggpairs(data)
# A  positive correlation exists for X1 and Y, X2 and Y, and X3 and Y 
# A negative correlation is displayed for X4 and Y
# All variables X1, X2, X3, X4 are significant for Y 

data_dff<-as.data.frame(sapply(data[-5], scale))#standardize the variables except the outcome variable Y
data_dff$Y<-as.numeric(data$Y)#add the Y column to data_dff

set.seed(1000) ## chosen arbitrarily; helps with replication across runs
data_dff_inTraining <- createDataPartition(data_dff$Y, ## Deaths is the outcome 
                                           p = .90, ## proportion used in training+ testing subset
                                           list = FALSE)

data_training_testing <- data_dff [data_dff_inTraining,]#contains 90% of the data

data_holdout<- data_dff[-data_dff_inTraining,]##contains 10% of the data

fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 3, ## k = 3. This represents the number of folds
                           repeats = 10)


# 2 Fit a baseline linear predictive model (MLR with lasso penalty), 
#where Y is the outcome, and the X variables are predictors. Use k-fold CV with multiple repetitions 
#(as feasible and appropriate), and compute and report the training error (RMSE)
set.seed(1000)

data_lasso <- train(Y~ ., # Predicting the outcome of Y using all other preds
                    data = data_training_testing, ## use the training_testing subset for data
                    method = "glmnet",
                    trControl = fitControl,
                    tuneGrid=expand.grid(alpha = 1, lambda = 1)) ## 1 means it is a lasso regression with a lambda being 1

data_lasso
#RMSE        Rsquared   MAE       
#5175669372  0.8939687  2326972132
data_lasso$results
summary(data_lasso)

plot(varImp(data_lasso))
#X1>X3>X2>X4

#3Next, fit a predictive model using the SVM approach and report the training error
data_gridsvm <- expand.grid(C = c(0.01, 0.1, 10))
#C is cost
data_preProcValues <- preProcess(data_training_testing, method = c("center", "scale"))


data_trainTransformed <- predict(data_preProcValues, data_training_testing) 
## apply the same scaling and centering on the holdout set, too
data_holdoutTransformed <- predict(data_preProcValues, data_holdout)


data_svmlinearfit <- train(Y ~ .,
                           data = data_trainTransformed, 
                           method = "svmLinear",
                           trControl = fitControl,
                           verbose = FALSE,
                           tuneGrid = data_gridsvm)

#C      RMSE       Rsquared   MAE 
#10.00  0.3523977  0.8929790  0.1385595 # This is lowest RMSE for this value

plot(varImp(data_svmlinearfit))
#X1>X3>X2>X4

#4 On the hold-out set, compute the prediction error, using the lasso MLR and SVM approaches. Comment on which approach performed better.
# Lasso Holdout set
set.seed(1000)

data_lasso_holdout <- train(Y~ ., # Predicting the outcome of Y using all other preds
                            data = data_holdout, ## use the training_testing subset for data
                            method = "glmnet",
                            trControl = fitControl,
                            tuneGrid=expand.grid(alpha = 1, lambda = 1))

data_lasso_holdout

#RMSE value is 5193697660

plot(varImp(data_lasso_holdout))
#X1>X2>X3>X4

data_svmlinearfit_holdout<- train(Y ~ .,
                                  data = data_holdoutTransformed, 
                                  method = "svmLinear",
                                  trControl = fitControl,
                                  verbose = FALSE,
                                  tuneGrid = data_gridsvm)

data_svmlinearfit_holdout
#C      RMSE       Rsquared   MAE    
#10.00  0.3557235  0.9010266  0.1446534
plot(varImp(data_svmlinearfit_holdout))
#X1>X3>X2>X4
#The RMSE value is better in SMV model for holdout

#5 Next, compute the variable importance based on the relative contribution made by each predictor towards the reduction of prediction error. 
#Here is an outline of such a procedure:

#5.1 Using the full model (i.e., including all of the predictors), compute the prediction error.

#5.2For each predictor in the dataset:

#5.2.1 Exclude the predictor,fit a model using all of the remaining predictors, and compute the prediction error on the hold out set.

#5.2.2 Save the difference in prediction errors made by the full model and the model excluding the current predictor.

#5.3 Rank the predictors on the basis of the absolute change in the prediction errors made between the full model and the model excluding the predictor.
#The greater the change in the prediction error made by including/excluding a predictor, greater is its relative importance.

#lasso X1,X2,X3

set.seed(1000)

data_lasso_holdout_without_x4 <- train(Y~ X1+X2+X3, # Predicting the outcome of Y using all other preds
                                       data = data_holdout, ## use the training_testing subset for data
                                       method = "glmnet",
                                       trControl = fitControl,
                                       tuneGrid=expand.grid(alpha = 1, lambda = 1))

data_lasso_holdout_without_x4
#RMSE        Rsquared   MAE       
#5861834487  0.8737067  3788334512
plot(varImp(data_lasso_holdout_without_x4))
#X1>X2>X3

#lasso X1,X2,X4

set.seed(1000)

data_lasso_holdout_without_x3 <- train(Y~ X1+X2+X4, # Predicting the outcome of Y using all other preds
                                       data = data_holdout, ## use the training_testing subset for data
                                       method = "glmnet",
                                       trControl = fitControl,
                                       tuneGrid=expand.grid(alpha = 1, lambda = 1))
data_lasso_holdout_without_x3
#RMSE        Rsquared   MAE       
#5660383179  0.8844309  2511962719
plot(varImp(data_lasso_holdout_without_x3))
#X1>X2>X4

#lasso X1,X3,X4

set.seed(1000)

data_lasso_holdout_without_x2 <- train(Y~ X1+X3+X4, # Predicting the outcome of Y using all other preds
                                       data = data_holdout, ## use the training_testing subset for data
                                       method = "glmnet",
                                       trControl = fitControl,
                                       tuneGrid=expand.grid(alpha = 1, lambda = 1))
data_lasso_holdout_without_x2
#RMSE        Rsquared   MAE       
#5713813504  0.8841643  2472691619
plot(varImp(data_lasso_holdout_without_x2))
#X1>X3>X4

#lasso X2,X3,X4

set.seed(1000)

data_lasso_holdout_without_x1 <- train(Y~ X2+X3+X4, # Predicting the outcome of Y using all other preds
                                       data = data_holdout, ## use the training_testing subset for data
                                       method = "glmnet",
                                       trControl = fitControl,
                                       tuneGrid=expand.grid(alpha = 1, lambda = 1))
data_lasso_holdout_without_x1
#RMSE        Rsquared   MAE       
#12792199465  0.4004059  4321556947
plot(varImp(data_lasso_holdout_without_x1))
#X2>X3>X4


#SVM X1,X2,X3

data_svmlinearfit_holdout_without_X4<- train(Y ~ X1+X2+X3,
                                             data = data_holdoutTransformed, 
                                             method = "svmLinear",
                                             trControl = fitControl,
                                             verbose = FALSE,
                                             tuneGrid = data_gridsvm)

data_svmlinearfit_holdout_without_X4
#C      RMSE       Rsquared   MAE    
#10.00  0.5187917  0.8596356  0.2011126
plot(varImp(data_svmlinearfit_holdout_without_X4))
#X1>X3>X2


#SVM X1,X2,X4

data_svmlinearfit_holdout_without_X3<- train(Y ~ X1+X2+X4,
                                             data = data_holdoutTransformed, 
                                             method = "svmLinear",
                                             trControl = fitControl,
                                             verbose = FALSE,
                                             tuneGrid = data_gridsvm)

data_svmlinearfit_holdout_without_X3
#C      RMSE       Rsquared   MAE    
#10.00  0.3882054  0.8825685  0.1487884
plot(varImp(data_svmlinearfit_holdout_without_X3))
#X1>X2>X4

#SVM X1,X3,X4

data_svmlinearfit_holdout_without_X2<- train(Y ~ X1+X3+X4,
                                             data = data_holdoutTransformed, 
                                             method = "svmLinear",
                                             trControl = fitControl,
                                             verbose = FALSE,
                                             tuneGrid = data_gridsvm)

data_svmlinearfit_holdout_without_X2
#C      RMSE       Rsquared   MAE    
#10.00 0.3868589  0.8804197  0.1465361
plot(varImp(data_svmlinearfit_holdout_without_X2))
#X1>X3>X4


#X2,X3,X4
data_svmlinearfit_holdout_without_X1<- train(Y ~ X2+X3+X4,
                                             data = data_holdoutTransformed, 
                                             method = "svmLinear",
                                             trControl = fitControl,
                                             verbose = FALSE,
                                             tuneGrid = data_gridsvm)

data_svmlinearfit_holdout_without_X1
#C      RMSE       Rsquared   MAE    
#10.00 0.8665818  0.3740480  0.2387773
plot(varImp(data_svmlinearfit_holdout_without_X1))
#X3>X2>X4

#5.3 Rank the predictors on the basis of the absolute change in the prediction errors made between 
#the full model and the model excluding the predictor. 
#The greater the change in the prediction error made by including/excluding a predictor, 
#greater is its relative importance.

#lasso 

resamps_LASSO<- resamples(list(fullholdout = data_lasso_holdout,
                               x1x2x3 = data_lasso_holdout_without_x4,
                               x1x2x4 = data_lasso_holdout_without_x3,
                               x1x3x4=data_lasso_holdout_without_x2,
                               x2x3x4=data_lasso_holdout_without_x1))


summary(resamps_LASSO)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps_LASSO, layout = c(3, 1))

#X1 is the most important in all the LASSO models. It contributed the most in reducing RMSE value. 
#In the full model holdout set ranking most important to least important was X1,X2,X3,X4 RMSE value  5193697660
#data_lasso_holdout_without_x4 ranking most important to least important was X1,X2,X3. RMSE value is 5838991926
#data_lasso_holdout_without_x3 ranking most important to least important was X1,X2,X4. RMSE value is 5660383179
#data_lasso_holdout_without_x2 ranking most important to least important was X1,X3,X4  RMSE value is 5713813504
#data_lasso_holdout_without_x1 ranking most important to least important was X2,X3,X4. RMSE value is 12792199465. The largest RMSE value. 
#With out X1 being used. It generated the largest error in the in the LASSO method



#SVM
resamps_SVM<- resamples(list(fullholdout = data_svmlinearfit_holdout,
                             x1x2x3 = data_svmlinearfit_holdout_without_X4,
                             x1x2x4 = data_svmlinearfit_holdout_without_X3,
                             x1x3x4=data_svmlinearfit_holdout_without_X2,
                             x2x3x4=data_svmlinearfit_holdout_without_X1))


summary(resamps_SVM)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps_SVM, layout = c(3, 1))
#X1 is the most important in all the SVM models. It contributed the most in reducing RMSE value.
#In the full model holdout set ranking most important to least important was X1,X3,X2,X4 RMSE value 0.3557235
#data_svmlinearfit_holdout_without_X4  most important to least important was X1,X2,X3    RMSE value 0.5187917
#data_svmlinearfit_holdout_without_X3  most important to least important was X1,X2,X4    RMSE value 0.3882054
#data_svmlinearfit_holdout_without_X2  most important to least important was X1,X3,X4    RMSE value 0.3868589
#data_svmlinearfit_holdout_without_X1  most important to least important was X3,X2,X4    RMSE value 0.8665818
#With out X1 being used. It generated the largest error in the in the SVM method

#6 After summarizing the variable importance values for each predictor across the two techniques (MLR + lasso and SVM), 
#comment on whether the relative importance of each variable predictor stays the same or changes, across the two techniques

#X1 remains to be the most important predictor in LASSO and SVM. The models that did not contain X1, had the highest error compared to the other models
#The models with out X2 came in second after the fullholdout in both LASSO and SVM 
#The model with out X3 came in third came in third for both.  
#The models with out X4 came in fourth in for both 
# The worse model performance was the once with out X1 as a predictor

#7 Did you standardize the variables prior to computing their variable importance? If you have, explain the rationale for doing so. 
#If you have not, explain the rationale for not doing so. 
#Ensure that your response includes a consideration of how the relative contribution of a predictor to the model 
#fit is affected by the scale on which each predictor is measured.

#Both models were standardized to make sure all the values were in the same unit of measurement. 
#The reason behind it was to make sure all the values were measured in their standard deviation score. 
#This would show what variable would have the most contribution as a predictor to the model it was going to fit.

