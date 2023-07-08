from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType


def main(streaming_directory: str) -> None:
    # Step 1. Create a SparkSession object
    spark_session = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("StreamingSelectParking") \
        .getOrCreate()

    logger = spark_session._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    # Step 2. Define a schema for the DataFrame object
    my_schema = StructType([
        StructField("poiID", StringType()),
        StructField("nombre", StringType()),
        StructField("direccion", StringType()),
        StructField("telefono", StringType()),
        StructField("correoelectronico", StringType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("altitud", DoubleType()),
        StructField("capacidad", IntegerType()),
        StructField("capacidad_discapacitados", IntegerType()),
        StructField("fechahora_ultima_actualizacion", TimestampType()),
        StructField("libres", IntegerType()),
        StructField("libres_discapacitados", IntegerType()),
        StructField("nivelocupacion_naranja", StringType()),
        StructField("nivelocupacion_rojo", StringType()),
        StructField("smassa_sector_sare", StringType())
    ])

    # Step 3. Read the dataframe from the streaming directory
    dataframe = spark_session \
        .readStream \
        .csv(streaming_directory, header=True, schema=my_schema)

    # Step 4. Filter (where) in the car parks whose field 'capacidad' is greater than 0
    # and select only three fields
    filtered_dataframe = dataframe \
        .filter(functions.col("capacidad") > 0) \
        .select("nombre", "capacidad", "libres")

    # Step 5. Start running the query that prints the output on the screen
    query = filtered_dataframe \
        .writeStream \
        .outputMode("update") \
        .format("console") \
        .start()

    query.awaitTermination()

    # Final Step. Stop SparkSession object
    spark_session.stop()


if __name__ == '__main__':
    # url = 'https://datosabiertos.malaga.eu/recursos/aparcamientos/ocupappublicosmun/ocupappublicosmun.csv'
    streaming_directory = './data/'
    # time_to_sleep = 15

    main(streaming_directory)
