# ensurePkg tests whether the packages that run_analysis uses are installed and, if not, installs them.
ensurePkg <- function(x) {
    if (!require(x,character.only = TRUE)) {
        install.packages(x,dep=TRUE, repos="http://cran.r-project.org")
        if(!require(x,character.only = TRUE)) stop("Package not found")
    }
}

ensurePkg("caret")
ensurePkg("ggplot2")
ensurePkg("dplyr")
ensurePkg("rattle")
ensurePkg("corrplot")
set.seed(42)

train <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
trainI <- createDataPartition(train$classe, p = 0.70, list=F)
training <- train[train, ]
validation <- train[-train, ]

# We are predicting a qualitative response variable and therefore will examine alternative classification
# machine learning models (logistic regression, random forest...). However, the response classes can be recast
# as binary (A and non-A), so linear regression will also be evaluated.

# Notes about the study (http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf):
#     1. There are 6 names in user_name, with wildly different counts
#     2. Imbalanced among user_name in testing group so can't afford to break down by user
#     3. Five "classes" of exercise form, with 'classe' A being correct
#     4. Four x-y-z sensors: belt, arm, forearm (glove), dumbbell
#     5. Sensors provide three-axes acceleration, gyroscope and magnetometer
#     6. For feature extraction we used a sliding window approach with different lengths from 0.5 second to 2.5 seconds, 
#        with 0.5 second overlap. In each step of the sliding window approach we calculated features on the Euler 
#        angles (roll, pitch and yaw), as well as the raw accelerometer, gyroscope and magnetometer readings. 
#        For the Euler angles of each of the four sensors we calculated eight features: mean, variance, standard 
#        deviation, max, min, amplitude, kurtosis and skew- ness, generating in total 96 derived feature sets.
#     7. They used random forest, using an ensemble of classifiers via Bagging.
#     8. Ten forests with ten trees, classifier tested with 10-fold cross-validation and different time window sizes
#     9. They also tried leave-one-out method


##Data Cleaning
# Lots of crap data, clean it up, too many with too few unique values
# avg*, stddev*, var*, min*, max* are all summary statistics and 100% NAs in the testing data
workTrain <- training
workTrain <- workTrain  %>% dplyr::select(-starts_with("avg"), 
                             -starts_with("stddev"),
                             -starts_with("var"), 
                             -starts_with("min"), 
                             -starts_with("max"),
                             -contains("timestamp"),
                             -contains("window"),
                             -X,
                             -user_name)

# Likewise, there are factors that have only "" and "#DIV/O" as factors:
workTrain <- workTrain %>% dplyr::select(-kurtosis_yaw_belt, 
                                         -skewness_yaw_belt, 
                                         -kurtosis_yaw_dumbbell, 
                                         -amplitude_yaw_belt, 
                                         -skewness_yaw_dumbbell, 
                                         -amplitude_yaw_dumbbell, 
                                         -kurtosis_yaw_forearm, 
                                         -skewness_yaw_forearm, 
                                         -amplitude_yaw_forearm)

#Remove zero covariates
nsv <- nearZeroVar(workTrain, saveMetrics = F)
workTrain <- workTrain[,-nsv]
#saveMetrics=T will give a data frame with frequency, %unique, actual zeroVar and nzv

#Remove variables with high proportion of missing data
propmiss <- function(dataframe) {
    m <- sapply(dataframe, function(x) {
        data.frame(
            nmiss=sum(is.na(x)), 
            n=length(x), 
            propmiss=sum(is.na(x))/length(x)
        )
    })
    d <- data.frame(t(m))
    d <- sapply(d, unlist)
    d <- as.data.frame(d)
    d$variable <- row.names(d)
    row.names(d) <- NULL
    d <- cbind(d[ncol(d)],d[-ncol(d)])
    return(d[order(d$propmiss), ])
}

missing <- propmiss(workTrain)
workTrain <- workTrain[,-as.integer(rownames(missing[missing$propmiss > 0.9,]))]

##Factor Analysis
# * Look for collinearity among variables
cor <- cor(workTrain[,-53])
corrplot.mixed(cor, order="AOE")


# * Prime factor analysis with all of them






##Logistic Regression
# * Binary classifier with A vs. B+C+D+E (why are they trying to predict WHICH incorrect form is being used..we only
#   care if the correct form is used)
# * With binary classifier, try logistic regression
binaryTrain <- workTrain
binaryTrain$classeBinary <- 0
binaryTrain$classeBinary[binaryTrain$classe == "A"] <- 1
binaryTrain$classe <- NULL
binaryTrain$classeBinary <- as.factor(binaryTrain$classeBinary)

# Train
fitLogi <- glm(classeBinary ~ ., data = binaryTrain, na.action = na.omit, control = list(maxit = 50), family = "binomial")
#Results in Warning messages:
#1: glm.fit: algorithm did not converge 
#2: glm.fit: fitted probabilities numerically 0 or 1 occurred 
# Which can be from a predictor that is nearly perfect, so categorization is better
# "linear separation"

fitGLM <- train(classeBinary ~ ., method="glm", data = binaryTrain)
fitGLM2 <- train(classeBinary ~ ., method="glm", data=binaryTrain, preProcess = c("center","scale"))

# Test
test <- predict(fitGLM, newdata=testing)

##Adaboost
#Train
#Test

##Random Forest
# Cross validation vs. other options because the dataset is sorted by classe so the train/test sets may be imbalanced
# Train
fit <- train(classe~., na.action = na.omit, data = training, method="rf") # 99.3% accuracy
fit2 <- train(classe~., na.action = na.omit, data = workTrain, method="rf") # 74% accuracy
fit3 <- train(classeBinary~., na.action = na.omit, data = binaryTrain, method="rf") # 89% accuracy
fitRF <- randomForest(classe ~ ., na.action=na.omit, ntree = 1000, importance = T, data=workTrain)
fitRFB <- randomForest(classeBinary ~ ., na.action=na.omit, ntree = 1000, importance = T, data=binaryTrain)
varImpPlot(fitRFB, main="Top 10 Most Important Predictors", n.var=10)
# final accuracy with all variables was 73.58%...not good enough
#mtry about 1000 crossed 70% threshold

# Test

##Model Comparison and Conclusion
predictions <- predict(whateverFit, newdata=validation)
confusionMatrix(predictions, validation$classe)
# * Include out-of-sample error estimation
