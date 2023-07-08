from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

if __name__ == '__main__':
    # Initial Step. Create a SparkSession object
    spark_session = SparkSession \
        .builder \
        .appName("RedWineCrossValidation") \
        .master("local[4]") \
        .getOrCreate()

    # Step 1. Read the CSV inferring the scheme
    data = spark_session \
        .read \
        .option("delimiter", ";") \
        .csv("C:\\Users\\jlpuente\\Documents\\MBD\\M10_Machine_Learning_y_Streaming\\Deliveries\\Advances\\2. "
             "Cross validation\\winequality-red.csv", header=True, inferSchema=True)

    # Step 2. Transform the read dataframe into another one with the format required by Spark ML library
    # Step 2.1 Get the column names of the input dataframe
    feature_cols = data.columns[:-1]
    label_col = data.columns[-1]

    # Step 2.2. Create a VectorAssembler object with the feature columns
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Step 2.3. Use the VectorAssembler to transform the input dataframe
    assembled_dataframe = assembler.transform(data)

    # Step 2.4. Select only the label column and the assembled features column
    string_indexer = StringIndexer(inputCol="quality", outputCol="label")
    si_model = string_indexer.fit(assembled_dataframe)
    dataframe = si_model \
        .transform(assembled_dataframe) \
        .select('features', 'label')

    dataframe.show()

    # Step 3. Select a multiclass classifier and train it using cross validation
    # Step 3.1. Split randomly the dataset into training and testing sets (75 % - 25 %)
    (training_data, test_data) = dataframe.randomSplit([0.75, 0.25])

    # Step 3.2. Define a Random Forest classifier
    random_forest_classifier = RandomForestClassifier()

    # Step 3.3. Define the parameter grid to search over during cross-validation
    param_grid = ParamGridBuilder() \
        .addGrid(random_forest_classifier.numTrees, [50, 100, 200]) \
        .addGrid(random_forest_classifier.maxDepth, [5, 10, 15, 20]) \
        .build()

    # Step 3.4. Define the evaluator for the multiclass classification problem
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    # Step 3.5. Define the cross-validator with 5 folds
    cross_validator = CrossValidator(estimator=random_forest_classifier,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     numFolds=5)

    # Step 3.6. Fit the model
    print('The model is starting to fit the model...')
    model = cross_validator.fit(training_data)
    print('The model has been fitted successfuly.')

    best_model = model.bestModel

    # Step 4. Analyze the quality of the classifier
    # Step 4.1. Make predictions on both the testing set and training set
    predictions_on_testing_set = best_model.transform(training_data)
    predictions = best_model.transform(test_data)

    # Step 4.2. Evaluate the accuracy of the predictions using MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy_on_training_set = evaluator.evaluate(predictions_on_testing_set)
    accuracy = evaluator.evaluate(predictions)

    # Step 4.3. Results and comments
    print(f'Best parameters: numTrees={best_model.getNumTrees}, maxDepth={best_model.getMaxDepth()}')
    print('Accuracy on testing set = {:.2f} %'.format(accuracy_on_training_set * 100))
    print('Accuracy on testing set = {:.2f} %'.format(accuracy * 100))

    print('''The Random Forest is the algorithm that better works on the red wine dataset (without dealing). From the 
    UCI repository they warns us about the classes are not balanced (e.g. there are many more normal wines than 
    excellent or poor ones). This means that hypothetically a model would generalize better the most represented 
    examples (that is, the normal wines, those whose label corresponds to quality 7, 6 and 5), but worse the minority 
    ones, those whose quality is either excellent (that is, 8) or poor (that is, 4 or 3). The result is indeed the 
    expected one and the model is overfitted.
    
    To break the sticking point at 68 % of accuracy on testing set, we must first apply some resampling or 
    oversampling (e.g. SMOTE) technique over the unbalanced dataset. Just applying this treatment on the dataset, 
    we may get accuracies near from 90 %.''')

    # Final Step. Stop SparkSession object
    spark_session.stop()
