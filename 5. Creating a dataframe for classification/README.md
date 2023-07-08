# Creating a dataframe for classification

We are interested in creating the following datataframe to work in a SparkML classification application:

| label | features           |
|-------|--------------------|
| 0.0   | [1.0, 3.0, 2.5]    |
| 1.0   | [0.0, -2.5, 1.7]   |
| 0.0   | [-3.0, -2.0, -1.0] |
| 1.0   | [3.6, -5.0, 2.5]   |

The goal of this exercise is to find at least three different ways to create it. An example is the following one:

## Way 1. From a RDD, no explicit scheme

```
data_frame = spark_session.sparkContext.parallelize(  
    [Row(label=0.0, features=Vectors.dense(1.0, 3.5, 4.2)),  
     Row(label=1.0, features=Vectors.dense(5.2, -2.1, 3.5)),  
     Row(label=0.0, features=Vectors.dense(-3, 2.4, 3.2)),  
     Row(label=1.0, features=Vectors.dense(3.6, -5.0, 2.5))]).toDF()  
  
data_frame = data_frame.select("label", "features")  
  
data_frame.printSchema()  
data_frame.show()
```
