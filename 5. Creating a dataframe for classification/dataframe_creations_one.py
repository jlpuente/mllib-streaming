from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


def main() -> None:

    data = [
        (0.0, Vectors.dense(1.0, 3.5, 4.2)),
        (1.0, Vectors.dense(5.2, -2.1, 3.5)),
        (0.0, Vectors.dense(-3, 2.4, 3.2)),
        (1.0, Vectors.dense(3.6, -5.0, 2.5))
    ]

    spark_session = SparkSession.builder.master("local[4]").getOrCreate()
    data_frame = spark_session.createDataFrame(data)
    data_frame.show()


if __name__ == '__main__':
    """DataFrame creation from a RDD"""
    main()
