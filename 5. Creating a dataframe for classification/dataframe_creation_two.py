import pandas as pd
from pyspark.sql import SparkSession


def main() -> None:

    data = {
        'label': [0.0, 1.0, 0.0, 1.0],
        'features': [[1.0, 3.5, 4.2], [5.2, -2.1, 3.5], [-3.0, 2.4, 3.2], [3.6, -5.0, 2.5]]
    }

    pandas_df = pd.DataFrame(data)
    spark_session = SparkSession.builder.master("local[2]").getOrCreate()
    dataframe = spark_session.createDataFrame(pandas_df)
    dataframe.show()


if __name__ == '__main__':
    """DataFrame creation from a Pandas DataFrame"""
    main()
