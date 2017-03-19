Preparing the data and R packages
=================================

Load packages, set caching
--------------------------

    library(caret)

    ## Warning: package 'caret' was built under R version 3.3.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(corrplot)

    ## Warning: package 'corrplot' was built under R version 3.3.3

    library(Rtsne)

    ## Warning: package 'Rtsne' was built under R version 3.3.3

    library(xgboost)

    ## Warning: package 'xgboost' was built under R version 3.3.3

    library(stats)
    library(knitr)
    library(ggplot2)
    library(rpart)

    ## Warning: package 'rpart' was built under R version 3.3.3

    library(rpart.plot)

    ## Warning: package 'rpart.plot' was built under R version 3.3.3

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 3.3.3

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    knitr::opts_chunk$set(cache=TRUE)

Getting data
------------

    # file names
    train.name = "C://Users//AnandVasumathi//Documents//pml-training.csv"
    test.name = "C://Users//AnandVasumathi//Documents//pml-testing.csv"

    # load the CSV files as data.frame 
    trainRaw = read.csv("C://Users//AnandVasumathi//Documents//pml-training.csv")
    testRaw = read.csv("C://Users//AnandVasumathi//Documents//pml-testing.csv")
    dim(trainRaw)

    ## [1] 19622   160

    dim(testRaw)

    ## [1]  20 160

    names(train)

    ## NULL

The raw training data has 19622 rows of observations and 158 features (predictors). Column X is unusable row number. While the testing data has 20 rows and the same 158 features. There is one column of target outcome named classe.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data Cleaning
-------------

First, extract target outcome (the activity quality) from training data, so now the training data contains only the predictors (the activity monitors).
-------------------------------------------------------------------------------------------------------------------------------------------------------

    sum(complete.cases(trainRaw))

    ## [1] 406

removing columns with NA
------------------------

    trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
    testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]

Finer Cleaning
--------------

    classe <- trainRaw$classe
    trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
    trainRaw <- trainRaw[, !trainRemove]
    trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
    trainCleaned$classe <- classe
    testRemove <- grepl("^X|timestamp|window", names(testRaw))
    testRaw <- testRaw[, !testRemove]
    testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

Slicing of Data
---------------

    set.seed(22519) # For reproducibile purpose
    inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
    trainData <- trainCleaned[inTrain, ]
    testData <- trainCleaned[-inTrain, ]

Modelling of data
-----------------

We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use 5-fold cross validation when applying the algorithm.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    controlRf <- trainControl(method="cv", 5)
    modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
    modelRf

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10989, 10991, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9901727  0.9875673
    ##   27    0.9917015  0.9895017
    ##   52    0.9840572  0.9798282
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

Then, we estimate the performance of the model on the validation data set.
--------------------------------------------------------------------------

    predictRf <- predict(modelRf, testData)
    confusionMatrix(testData$classe, predictRf)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    0    0    0    1
    ##          B    5 1131    3    0    0
    ##          C    0    0 1021    5    0
    ##          D    0    0   13  949    2
    ##          E    0    0    1    6 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9939          
    ##                  95% CI : (0.9915, 0.9957)
    ##     No Information Rate : 0.2851          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9923          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9970   1.0000   0.9836   0.9885   0.9972
    ## Specificity            0.9998   0.9983   0.9990   0.9970   0.9985
    ## Pos Pred Value         0.9994   0.9930   0.9951   0.9844   0.9935
    ## Neg Pred Value         0.9988   1.0000   0.9965   0.9978   0.9994
    ## Prevalence             0.2851   0.1922   0.1764   0.1631   0.1832
    ## Detection Rate         0.2843   0.1922   0.1735   0.1613   0.1827
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9984   0.9992   0.9913   0.9927   0.9979

The average accuracy is 99.4%, with error rate is 0.60%. So, expected error rate of less than 1% is fulfilled
-------------------------------------------------------------------------------------------------------------

    accuracy <- postResample(predictRf, testData$classe)
    accuracy

    ##  Accuracy     Kappa 
    ## 0.9938828 0.9922620

    oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
    oose

    ## [1] 0.006117247

Predicting for Test Data Set
============================

Now, we apply the model to the original testing data set downloaded from the data source. We remove the problem\_id column first.
---------------------------------------------------------------------------------------------------------------------------------

    result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
    result

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Appendix: Figures
=================

Correlation Matrix Visualization
--------------------------------

Plot of correlation matrix
--------------------------

Plot a correlation matrix between features.
-------------------------------------------

A good set of features is when they are highly uncorrelated (orthogonal) each others. The plot below shows average of correlation is not too high, so I choose to not perform further PCA preprocessing.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    corrPlot <- cor(trainData[, -length(names(trainData))])
    corrplot(corrPlot, method="color")

![](PML_Project_report_2_files/figure-markdown_strict/unnamed-chunk-11-1.png)
\#\# Decision Tree Visualization

    treeModel <- rpart(classe ~ ., data=trainData, method="class")
    prp(treeModel) # fast plot

![](PML_Project_report_2_files/figure-markdown_strict/unnamed-chunk-12-1.png)
