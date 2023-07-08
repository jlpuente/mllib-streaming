from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


if __name__ == '__main__':

    # Step 1. Create a SparkSession object
    spark_session = SparkSession \
        .builder \
        .appName("IrisDataset") \
        .master("local[2]") \
        .getOrCreate()

    # Step 2. Load the wine dataset into a Spark DataFrame
    data = spark_session\
        .read\
        .csv("C:\\Users\\jlpuente\\Documents\\MBD\\M10_Machine_Learning_y_Streaming\\Deliveries\\Mandatories\\4. "
             "Classification wine dataset\\wines", header=False, inferSchema=False)

    # Step 3. Convert columns to float data type
    for col in data.columns:
        data = data.withColumn(col, data[col].cast("float"))

    # Step 4. Embrace the feature columns in one single feature column
    num_features = len(data.columns) - 1
    feature_cols = ["feature_" + str(index) for index in range(1, num_features + 1)]

    # Step 5. Define the vector assembler to create a "features" column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Step 6. Add a header to the wine dataframe
    data = data.toDF("label", *feature_cols)

    # Step 7. Transform the wine dataframe into a new one with the new header by using the vector assembler
    wines_dataframe = assembler\
        .transform(data)\
        .select("label", "features")

    # Step 8. Split randomly the dataset into training and testing sets (75 % - 25 %)
    (training_data, test_data) = wines_dataframe.randomSplit([0.75, 0.25])

    # Step 9. Define a scaler to normalize the feature vector for zero mean and unit standard deviation
    normalizer = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    # Step 10. Define the random forest classifier
    random_forest_classifier = RandomForestClassifier(numTrees=10, maxDepth=5, seed=42, labelCol="label", featuresCol="scaledFeatures")

    # Step 11. Create a pipeline
    pipeline = Pipeline(stages=[normalizer, random_forest_classifier])

    # Step 12. Fit the pipeline to the training data by fitting and transforming the stages inside
    model = pipeline.fit(training_data)

    # Step 13. Make predictions on the testing data
    predictions = model.transform(test_data)

    # Step 14. Evaluate the accuracy of the model
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f'The accuracy of the model is: {round(accuracy * 100, 2)} %')

    # Final Step. Stop SparkSession object
    spark_session.stop()
