# Data preprocessing and crossvalidation

This exercise is about applying the  [cross validation feature of Spark ML](https://spark.apache.org/docs/latest/ml-tuning.html#cross-validation) when dealing with the classification of the red  [wine quality data set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  included in the UCI repository.

  

The assignment consists of writting a Python program called _wineCrossvalidation.py_ which must do the following:

1.  Read the CSV inferring the scheme.
2.  Transforming the read dataframe into another one with the format required by SparkML.
3.  Select a classifier and train it using crossvalidation.
4.  Analyze the quality of the classifier.  

Requirements:

-   The program is intended to be prepared to be used with a huge amount of data, so Steps 1 and 2 must use SparkML features (that is, preprocessing the data file with Pandas is not allowed).
-   You are free to select the classification algorithm.

Deliverable:

-   The **.py** program.
-   A **screenshoot** showing the program output about the analysis of the quality of the classifier.
