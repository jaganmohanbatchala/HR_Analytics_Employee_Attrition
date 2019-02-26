## *************************************************** ##
#  Author : Jagan Mohan Batchala
#  Problem statement : IBM HR Analytics Employee Attrition & Performance
#  Hypothesis : Employee Attrition
## *************************************************** ##

## clear the global environments
rm(list = ls(all = TRUE))

## load required libraries
library(ROCR)  # to plot the ROC curve
library(DMwR)  # used for missing values imputation and for SMOTE technique
library(dplyr)
library(caret) # used for data partition
library(ROCR)  # to plot the ROC curve
library(caret) # used for confusion matrix
library(car)   # used for vif to find multi-collinearity
library(MASS)  # Improve the model using stepAIC
library(e1071) # to build naiveBayes
library(dummies)# to create dummy variables
library(class)
library(rpart) # used for decision tree
library(rpart.plot) # used for decision tree
library(randomForest) # used for Random Forest model
library(C50)  # used to build Decision Tree 

## set working directory
dir_path ="../Desktop/INTERN/HumanResourceManagement"
setwd(dir_path)

## load dataset
raw_data <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", header = TRUE)

## EXploratory Data Analysis - Understand the data
cat("Given data Records :", dim(raw_data)[1], ", Attributes :  ", dim(raw_data)[2])
head(raw_data,5)
tail(raw_data,5)
colnames(raw_data)
summary(raw_data)
str(raw_data)
sort(colSums(is.na(raw_data)), decreasing = TRUE)
sapply(raw_data, class)

## check levels for each attribute
sapply(raw_data, levels)
apply(raw_data,2, function(x) {table(x)})

head(raw_data[,c(9,10,22,27)],5)
tail(raw_data[,c(9,10,22,27)],5)

p1_preprocessing <- raw_data
sum(is.na(duplicated(p1_preprocessing)))

## drop the columns : EmployeeCount , EmployeeNumber , Over18 , StandardHours
p1_preprocessing <- p1_preprocessing[,-c(9,10,22,27)]
dim(p1_preprocessing)

## levels attributes convert to categorical
## Identified attributes : Education , JObLevel , 
table(raw_data$Education)
raw_data[raw_data$JObLevel]
table(raw_data$JobSatisfaction)
table(raw_data$PerformanceRating)
table(raw_data$RelationshipSatisfaction)
table(raw_data$StockOptionLevel)
table(raw_data$WorkLifeBalance)

names(p1_preprocessing$ï..Age) <-  "Age"

p2_preprocessing <- p1_preprocessing

p2_preprocessing$Education <- as.factor(as.character(p2_preprocessing$Education))
p2_preprocessing$JobLevel <- as.factor(as.character(p2_preprocessing$JobLevel)) 
p2_preprocessing$JobSatisfaction <- as.factor(as.character(p2_preprocessing$JobSatisfaction)) 
p2_preprocessing$PerformanceRating <- as.factor(as.character(p2_preprocessing$PerformanceRating)) 
p2_preprocessing$RelationshipSatisfaction <- as.factor(as.character(p2_preprocessing$RelationshipSatisfaction)) 
p2_preprocessing$StockOptionLevel <- as.factor(as.character(p2_preprocessing$StockOptionLevel)) 
p2_preprocessing$WorkLifeBalance <- as.factor(as.character(p2_preprocessing$WorkLifeBalance)) 

str(p2_preprocessing)

num_p1data <- p2_preprocessing[ , sapply(p2_preprocessing, is.numeric)]
cat_p1data <- p2_preprocessing[ , !sapply(p2_preprocessing, is.numeric)]

str(num_p1data)
str(cat_p1data)
summary(num_p1data)

cat_p1data <- apply(cat_p1data, 2, function(x){as.factor(as.character(x))})
num_p1data <- apply(num_p1data, 2, function(x){as.numeric(as.character(x))})

p2_preprocessing <- cbind.data.frame(num_p1data, cat_p1data)
names(p2_preprocessing)
ncol(p2_preprocessing)

p3_preprocessing <- p2_preprocessing



## Identified target variable 
## Modify No,Yes values into 0,1 and convert into categorical variable
table(p3_preprocessing$Attrition)
p3_preprocessing$Attrition <- ifelse(p3_preprocessing$Attrition == 'Yes',1,0)

cor(num_p1data)
## MonthlyIncome variable is highly correlated (0.9503)with JobLevel and should be ignored for analysis
p3_preprocessing$MonthlyIncome <- NULL

## load DMwR package
table(p3_preprocessing$Attrition)
p3_preprocessing$Attrition <- as.factor(p3_preprocessing$Attrition)
p3_preprocessing <- SMOTE(Attrition ~ ., p3_preprocessing, 
                          perc.over = 200, perc.under = 100, 
                          k=5, learner = NULL)
table(p3_preprocessing$Attrition)


## split given data into train and test
## Use createDataPartition to create stratified sampling
set.seed(123)
index <- createDataPartition(p3_preprocessing$Attrition, p = 0.75, list = F)
#index <- createDataPartition(p3_preprocessing$Attrition, p = 0.9, list = F)
traindata <- p3_preprocessing[index, ]
testdata <- p3_preprocessing[-index, ]

## ************************************************************************************* ##
## ***************** build the classification model : NaiveBayes  ************** ##
## ************************************************************************************* ##
model_nb <- naiveBayes(traindata$Attrition ~ ., data = traindata)
model_nb

# prediction using model on train data
pred_train_nb <- predict(model_nb, newdata = traindata)
conf_matrix_train_nb <- table(pred_train_nb, traindata$Attrition)
print(conf_matrix_train_nb)

specificity_train_nb <- round(conf_matrix_train_nb[1,1]/sum(conf_matrix_train_nb[1,]),4)
sensitivity_train_nb <- round(conf_matrix_train_nb[2,2]/sum(conf_matrix_train_nb[2,]),4)
precision_train_nb <- round(conf_matrix_train_nb[2,2]/sum(conf_matrix_train_nb[,2]),4)
accuracy_train_nb <- round(sum(diag(conf_matrix_train_nb))/sum(conf_matrix_train_nb),4)

# prediction using model on test data
pred_test_nb <- predict(model_nb, newdata = testdata)
conf_matrix_test_nb <- table(pred_test_nb, testdata$Attrition)
print(conf_matrix_test_nb)

specificity_test_nb <- round(conf_matrix_test_nb[1,1]/sum(conf_matrix_test_nb[1,]),4)
sensitivity_test_nb <- round(conf_matrix_test_nb[2,2]/sum(conf_matrix_test_nb[2,]),4)
precision_test_nb <- round(conf_matrix_test_nb[2,2]/sum(conf_matrix_test_nb[,2]),4)
accuracy_test_nb <- round(sum(diag(conf_matrix_test_nb))/sum(conf_matrix_test_nb),4)

cat("Accuracy :",accuracy_train_nb*100,"\nSpecificity :",specificity_train_nb*100,"\nSensitivity:",sensitivity_train_nb*100,"\nPrecision:",precision_train_nb*100)
cat("Accuracy :",accuracy_test_nb*100,"\nSpecificity :",specificity_test_nb*100,"\nSensitivity:",sensitivity_test_nb*100,"\nPrecision:",precision_test_nb*100)

confusionMatrix(pred_train_nb, traindata$Attrition, positive = "1")
confusionMatrix(pred_test_nb, testdata$Attrition, positive = "1")

## ************************************************************************************* ##
## ***************** build the classification model : Logistic Regression ************** ##
## ************************************************************************************* ##

model_glm <- glm(formula = Attrition~., data = traindata, family = binomial)
summary(model_glm)

# prediction using model
prob_train <- predict(model_glm, newdata = traindata, type = "response")
prob_test <- predict(model_glm, newdata = testdata, type = "response") # Predicting on test data

## calculate auc by plotting ROC curve
## load ROCR package
pred_train <- prediction(prob_train, traindata$Attrition)
perf_train <- performance(pred_train, measure = "tpr", x.measure = "fpr")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf_train, col = rainbow(10), colorize=T, 
     main = "ROC Curve", ylab = "Sensitivity (TPR) ",
     xlab = "Specificity (FPR) ",
     print.cutoffs.at = seq(0,1,0.05))
perform_auc <- performance(pred_train, measure="auc")
#abline(a = 0, b =1)
auc <- perform_auc@y.values[[1]]
auc <- round(auc,2)
legend(.4, .7, auc, title="AUC")
print(auc)

preds_train <- ifelse(prob_train > 0.5, 1, 0)
conf_matrix_train_glm <- table(traindata$Attrition, preds_train)
print(conf_matrix_train_glm)
specificity_train_glm <- conf_matrix_train_glm[1, 1]/sum(conf_matrix_train_glm[1, ])
sensitivity_train_glm <- conf_matrix_train_glm[2, 2]/sum(conf_matrix_train_glm[2, ])
precision_train_glm <- round(conf_matrix_train_glm[2,2]/sum(conf_matrix_train_glm[,2]),4)
accuracy_train_glm <- sum(diag(conf_matrix_train_glm))/sum(conf_matrix_train_glm)

preds_test <- ifelse(prob_test > 0.5, 1, 0)
conf_matrix_test_glm <- table(testdata$Attrition, preds_test)
print(conf_matrix_test_glm)
specificity_test_glm <- conf_matrix_test_glm[1, 1]/sum(conf_matrix_test_glm[1, ])
sensitivity_test_glm <- conf_matrix_test_glm[2, 2]/sum(conf_matrix_test_glm[2, ])
precision_test_glm <- round(conf_matrix_test_glm[2,2]/sum(conf_matrix_test_glm[,2]),4)
accuracy_test_glm <- sum(diag(conf_matrix_test_glm))/sum(conf_matrix_test_glm)

cat("Accuracy :",accuracy_train_glm*100,"\nSpecificity :",specificity_train_glm*100,"\nSensitivity:",sensitivity_train_glm*100,"\nPrecision:",precision_train_glm*100)
cat("Accuracy :",accuracy_test_glm*100,"\nSpecificity :",specificity_test_glm*100,"\nSensitivity:",sensitivity_test_glm*100,"\nPrecision:",precision_test_glm*100)

confusionMatrix(preds_train, traindata$Attrition, positive = "1")
confusionMatrix(preds_test, testdata$Attrition, positive = "1")

## ************************************************************************************* ##
## ***************** build the classification model : Random Forest       ************** ##
## ************************************************************************************* ##
## load randomforest package
model_RF <- randomForest(Attrition ~., data = traindata)
model_RF
pred_RF_train <- predict(model_RF, newdata = traindata, type ="class")
pred_RF_test <- predict(model_RF, newdata = testdata, type ="class")

conf_matrix_train_RF <- table(traindata$Attrition, pred_RF_train)
conf_matrix_test_RF <- table(testdata$Attrition, pred_RF_test)
print(conf_matrix_train_RF)
print(conf_matrix_test_RF)

specificity_train_RF <- conf_matrix_train_RF[1, 1]/sum(conf_matrix_train_RF[1, ])
sensitivity_train_RF <- conf_matrix_train_RF[2, 2]/sum(conf_matrix_train_RF[2, ])
precision_train_RF <- round(conf_matrix_train_RF[2,2]/sum(conf_matrix_train_RF[,2]),4)
accuracy_train_RF <- sum(diag(conf_matrix_train_RF))/sum(conf_matrix_train_RF)

conf_matrix_test_RF <- table(testdata$Attrition, pred_RF_test)
print(conf_matrix_test_RF)
specificity_test_RF <- conf_matrix_test_RF[1, 1]/sum(conf_matrix_test_RF[1, ])
sensitivity_test_RF <- conf_matrix_test_RF[2, 2]/sum(conf_matrix_test_RF[2, ])
precision_test_RF <- round(conf_matrix_test_RF[2,2]/sum(conf_matrix_test_RF[,2]),4)
accuracy_test_RF <- sum(diag(conf_matrix_test_RF))/sum(conf_matrix_test_RF)

cat("Accuracy :",accuracy_train_RF*100,"\nSpecificity :",specificity_train_RF*100,"\nSensitivity:",sensitivity_train_RF*100,"\nPrecision:",precision_train_RF*100)
cat("Accuracy :",accuracy_test_RF*100,"\nSpecificity :",specificity_test_RF*100,"\nSensitivity:",sensitivity_test_RF*100,"\nPrecision:",precision_test_RF*100)

confusionMatrix(pred_RF_train, traindata$Attrition, positive = "1")
confusionMatrix(pred_RF_test, testdata$Attrition, positive = "1")

## ************************************************************************************* ##
## ***************** build the classification model : Decision Tree       ************** ##
## ************************************************************************************* ##
## load rpart package

model_DT <- rpart(Attrition ~., data = traindata, method = "class")    
print(model_DT)
plot(model_DT, main = "Classification Tree for Attrition Class", margin = 0.15, uniform = TRUE)
text(model_DT, use.n = T)
summary(model_DT)


rpart.plot(model_DT, fallen.leaves = T)  # plotting rpart plot

## predict 
pred_train <- predict(model_DT, newdata = traindata, type = "class")
pred_test <- predict(model_DT, newdata = testdata, type = "class")

conf_matrix_train_DT <- table(traindata$Attrition, pred_train)
conf_matrix_test_DT <- table(testdata$Attrition, pred_test)
print(conf_matrix_train_DT)
print(conf_matrix_test_DT)

specificity_train_DT <- conf_matrix_train_DT[1, 1]/sum(conf_matrix_train_DT[1, ])
sensitivity_train_DT <- conf_matrix_train_DT[2, 2]/sum(conf_matrix_train_DT[2, ])
precision_train_DT <- round(conf_matrix_train_DT[2,2]/sum(conf_matrix_train_DT[,2]),4)
accuracy_train_DT <- sum(diag(conf_matrix_train_DT))/sum(conf_matrix_train_DT)

specificity_test_DT <- conf_matrix_test_DT[1, 1]/sum(conf_matrix_test_DT[1, ])
sensitivity_test_DT <- conf_matrix_test_DT[2, 2]/sum(conf_matrix_test_DT[2, ])
precision_test_DT <- round(conf_matrix_test_DT[2,2]/sum(conf_matrix_test_DT[,2]),4)
accuracy_test_DT <- sum(diag(conf_matrix_test_DT))/sum(conf_matrix_test_DT)

cat("Accuracy :",accuracy_train_DT*100,"\nSpecificity :",specificity_train_DT*100,"\nSensitivity:",sensitivity_train_DT*100,"\nPrecision:",precision_train_DT*100)
cat("Accuracy :",accuracy_test_DT*100,"\nSpecificity :",specificity_test_DT*100,"\nSensitivity:",sensitivity_test_DT*100,"\nPrecision:",precision_test_DT*100)

## Check with Complexity Parameter of the tree with cp=0
model_DT_CP <- rpart(Attrition ~ ., data = traindata, method = "class", cp = 0.02)
printcp(model_DT_CP)

pred_train <- predict(model_DT_CP, newdata = traindata, type = "class")
pred_test <- predict(model_DT_CP, newdata = testdata, type = "class")

conf_matrix_train_DT_CP <- table(traindata$Attrition, pred_train)
conf_matrix_test_DT_CP <- table(testdata$Attrition, pred_test)
print(conf_matrix_train_DT_CP)
print(conf_matrix_test_DT_CP)

specificity_train_DT_CP <- conf_matrix_train_DT_CP[1, 1]/sum(conf_matrix_train_DT_CP[1, ])
sensitivity_train_DT_CP <- conf_matrix_train_DT_CP[2, 2]/sum(conf_matrix_train_DT_CP[2, ])
precision_train_DT_CP <- round(conf_matrix_train_DT_CP[2,2]/sum(conf_matrix_train_DT_CP[,2]),4)
accuracy_train_DT_CP <- sum(diag(conf_matrix_train_DT_CP))/sum(conf_matrix_train_DT_CP)

specificity_test_DT_CP <- conf_matrix_test_DT_CP[1, 1]/sum(conf_matrix_test_DT_CP[1, ])
sensitivity_test_DT_CP <- conf_matrix_test_DT_CP[2, 2]/sum(conf_matrix_test_DT_CP[2, ])
precision_test_DT_CP <- round(conf_matrix_test_DT_CP[2,2]/sum(conf_matrix_test_DT_CP[,2]),4)
accuracy_test_DT_CP <- sum(diag(conf_matrix_test_DT_CP))/sum(conf_matrix_test_DT_CP)

cat("Accuracy :",accuracy_train_DT_CP*100,"\nSpecificity :",specificity_train_DT_CP*100,"\nSensitivity:",sensitivity_train_DT_CP*100,"\nPrecision:",precision_train_DT_CP*100)
cat("Accuracy :",accuracy_test_DT_CP*100,"\nSpecificity :",specificity_test_DT_CP*100,"\nSensitivity:",sensitivity_test_DT_CP*100,"\nPrecision:",precision_test_DT_CP*100)

confusionMatrix(pred_train, traindata$Attrition, positive = "1")
confusionMatrix(pred_test, testdata$Attrition, positive = "1")
## ************************************************************************************* ##
## *****************  Descision Tree - C5.0    ***************************************** ##
## ************************************************************************************* ##
model_DTC5 <- C5.0(Attrition ~ ., data = traindata, rules = TRUE)
summary(model_DTC5)
C5imp(model_DTC5, pct = TRUE)

pred_train <- predict(model_DTC5, newdata = traindata, type = "class")
pred_test <- predict(model_DTC5, newdata = testdata, type = "class")
conf_matrix_train_DTC5 <- table(traindata$Attrition, pred_train)
conf_matrix_test_DTC5 <- table(testdata$Attrition, pred_test)
print(conf_matrix_train_DTC5)
print(conf_matrix_test_DTC5)

specificity_train_DTC5 <- conf_matrix_train_DTC5[1, 1]/sum(conf_matrix_train_DTC5[1, ])
sensitivity_train_DTC5 <- conf_matrix_train_DTC5[2, 2]/sum(conf_matrix_train_DTC5[2, ])
precision_train_DTC5 <- round(conf_matrix_train_DTC5[2,2]/sum(conf_matrix_train_DTC5[,2]),4)
accuracy_train_DTC5 <- sum(diag(conf_matrix_train_DTC5))/sum(conf_matrix_train_DTC5)

specificity_test_DTC5 <- conf_matrix_test_DTC5[1, 1]/sum(conf_matrix_test_DTC5[1, ])
sensitivity_test_DTC5 <- conf_matrix_test_DTC5[2, 2]/sum(conf_matrix_test_DTC5[2, ])
precision_test_DTC5 <- round(conf_matrix_train_DTC5[2,2]/sum(conf_matrix_train_DTC5[,2]),4)
accuracy_test_DTC5 <- sum(diag(conf_matrix_test_DTC5))/sum(conf_matrix_test_DTC5)

cat("Accuracy :",accuracy_train_DTC5*100,"\nSpecificity :",specificity_train_DTC5*100,"\nSensitivity:",sensitivity_train_DTC5*100,"\nPrecision:",precision_train_DT_CP*100)
cat("Accuracy :",accuracy_test_DTC5*100,"\nSpecificity :",specificity_test_DTC5*100,"\nSensitivity:",sensitivity_test_DTC5*100,"\nPrecision:",precision_test_DT_CP*100)

confusionMatrix(pred_train, traindata$Attrition, positive = "1")
confusionMatrix(pred_test, testdata$Attrition, positive = "1")

# ************************************************************************************* ##
## *****************  build the classification model : SVM   ***************************************** ##
## ************************************************************************************* ##

model_svm_linear <- svm(Attrition ~.,data = traindata, kernel = "linear", cost = 10)
summary(model_svm_linear)

pred_train_linear <-table(predict(model_svm_linear), traindata$Attrition)
confusionMatrix(pred_train_linear, positive = "1")

model_svm_radial <- svm(Attrition ~.,data = traindata, kernel = "radial", cost = 10)
summary(model_svm_linear)

pred_train_radial <-table(predict(model_svm_radial), traindata$Attrition)
confusionMatrix(pred_train_radial, positive = "1")

## Tuning parameters for SVM
model_svm_tuned <- tune.svm(Attrition ~., data = traindata, gamma = 10^(-6:-1), cost = 10^(1:5))
summary(model_svm_tuned)

svmfit <- svm (Attrition ~ ., data = traindata, kernel = "radial", cost = 10, gamma = 0.01)
print(svmfit)

pred_train_fit <-table(predict(svmfit), traindata$Attrition)
confusionMatrix(pred_train_fit, positive = "1")

model_svm_gamma <- svm (Attrition ~ ., data = traindata, kernel = "radial", cost = 10, gamma = 0.021)
print(model_svm_gamma)

pred_train_gamma <-table(predict(model_svm_gamma), traindata$Attrition)
confusionMatrix(pred_train_gamma, positive = "1")

model_svm_gamma_test <- svm (Attrition ~ ., data = testdata, kernel = "radial", cost = 10, gamma = 0.021)
print(model_svm_gamma_test)

pred_test_gamma <-table(predict(model_svm_gamma_test), testdata$Attrition)
confusionMatrix(pred_test_gamma, positive = "1")

conf_matrix_train_svm <- pred_train_gamma 
conf_matrix_test_svm <- pred_test_gamma 
print(conf_matrix_train_svm)
print(conf_matrix_test_svm)

specificity_train_svm <- conf_matrix_train_svm[1, 1]/sum(conf_matrix_train_svm[1, ])
sensitivity_train_svm <- conf_matrix_train_svm[2, 2]/sum(conf_matrix_train_svm[2, ])
precision_train_svm <- round(conf_matrix_train_svm[2,2]/sum(conf_matrix_train_svm[,2]),4)
accuracy_train_svm <- sum(diag(conf_matrix_train_svm))/sum(conf_matrix_train_svm)

specificity_test_svm <- conf_matrix_test_svm[1, 1]/sum(conf_matrix_test_svm[1, ])
sensitivity_test_svm <- conf_matrix_test_svm[2, 2]/sum(conf_matrix_test_svm[2, ])
precision_test_svm <- round(conf_matrix_test_svm[2,2]/sum(conf_matrix_test_svm[,2]),4)
accuracy_test_svm <- sum(diag(conf_matrix_test_svm))/sum(conf_matrix_test_svm)

cat("Accuracy :",accuracy_train_svm*100,"\nSpecificity :",specificity_train_svm*100,"\nSensitivity:",sensitivity_train_svm*100,"\nPrecision:",precision_train_svm*100)
cat("Accuracy :",accuracy_test_svm*100,"\nSpecificity :",specificity_test_svm*100,"\nSensitivity:",sensitivity_test_svm*100,"\nPrecision:",precision_test_svm*100)




