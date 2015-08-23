library(caret)
library(parallel)
library(doParallel)
#do parallel computing using multicore
registerDoParallel(makeCluster(detectCores()))

#load raw data
training <- read.csv("data/pml-training.csv", header = TRUE)
test  <- read.csv('data/pml-testing.csv', header = TRUE)


#### explore data
dim(training)
dim(test)

table(training$classe)

########clean ####

#remove columns with over a 80% of not a number
naCol<- apply(training,2,function(x) {sum(is.na(x))});
training <- training[,which(naCol <  nrow(training)*0.8)];  
dim(training)

#remove near zero variance predictors
nz <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nz$nzv==FALSE]
dim(training)

#remove not relevant columns for classification 
#(x, user_name, raw time stamp 1  and 2, "new_window" and "num_window")

removeIndex<- grep("timestamp|X|user_name|window",names(training))
training <- training[,-removeIndex]
dim(training)
#class into factor
training$classe <- factor(training$classe)


########  Split the data: 80% for training, 20% for testing
trainIndex <- createDataPartition(y = training$classe, p=0.8,list=FALSE)
trainingSample <- training[trainIndex,]
testingSample <- training[-trainIndex,]
dim(trainingSample)
dim(testingSample)

############ Create machine learning models
#random seed
set.seed(8143)
#random forest   ("rf")
#boosted trees ("gbm") 
model_rf <- train(classe ~ .,  method="rf", data=trainingSample)    
model_gbm <-train(classe ~ ., method = 'gbm', data = trainingSample)


# Scoring - Confusion matrix
print("Random forest  ")
rf_predict<- predict(model_rf, testingSample)
print(confusionMatrix(rf_predict, testingSample$classe))

print("Boosted trees GBM ")
gbm_predict<- predict(model_gbm , testingSample)
print(confusionMatrix(gbm_predict, testingSample$classe))


#Cross validation and tuning
#random seed
set.seed(8143)
#parallel computing for multi-core
registerDoParallel(makeCluster(detectCores()))

cv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
model_rf_CV <- train(classe ~ ., method="rf",  data=trainingSample, trControl = cv_control)

#final accuracy
print("Random forest accuracy after Cross validation")
rf_CV_accuracy<<- predict(model_rf_CV , testingSample)
print(confusionMatrix(rf_CV_accuracy, testingSample$classe))

#Important Variables 
print("Variables importance")
vi = varImp(model_rf_CV$finalModel)
vi$var<-rownames(vi)
vi = as.data.frame(vi[with(vi, order(vi$Overall, decreasing=TRUE)), ])
rownames(vi) <- NULL
print(vi)

#predict
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


#Prediction
saveOut<- function(){
  prediction <- predict(model_rf_CV, test)
  print(prediction)
  answers <- as.vector(prediction)
  pml_write_files(answers)
}
#dump output
saveOut()
