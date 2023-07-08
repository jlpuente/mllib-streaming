from pyspark.sql import SparkSession


def main() -> None:

    data = [
        (0.0, [1.0, 3.5, 4.2]),
        (1.0, [5.2, -2.1, 3.5]),
        (0.0, [-3.0, 2.4, 3.2]),
        (1.0, [3.6, -5.0, 2.5])
    ]

    spark_session = SparkSession.builder.master("local[2]").getOrCreate()
    dataframe = spark_session.createDataFrame(data, ["label", "features"])
    dataframe.show()


if __name__ == '__main__':
    """DataFrame creation from a list of tuples"""
    main()
