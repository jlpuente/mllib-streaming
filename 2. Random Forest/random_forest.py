from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


if __name__ == '__main__':

    # Step 1. Create a SparkSession
    spark_session = SparkSession\
        .builder\
        .appName("RandomForestBreastCancer")\
        .master("local[4]")\
        .getOrCreate()

    # Step 2. Load breast cancer dataset, in its scaled version and LIBSVM format
    data = spark_session\
        .read\
        .format("libsvm")\
        .load("C:\\Users\\jlpuente\\Documents\\MBD\\M10_Machine_Learning_y_Streaming\\Deliveries\\Mandatories\\2. "
              "Random forest\\breast-cancer_scale.txt")

    # Step 3. Split data into training (75 %) and testing (25 %) sets
    (training_set, test_set) = data.randomSplit([0.75, 0.25], seed=42)

    # Step 4. Train a Random Forest classifier
    random_forest_classifier = RandomForestClassifier(numTrees=10, maxDepth=5, seed=42)
    model = random_forest_classifier.fit(training_set)

    # Step 5. Make predictions on the testing set
    predictions = model.transform(test_set)

    # Step 6. Evaluate the accuracy of the model using a multiclass classification evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f'Accuracy = {round(accuracy * 100, 2)} %')

    # Final Step. Stop the SparkSession object
    spark_session.stop()
