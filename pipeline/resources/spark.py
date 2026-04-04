from dagster import ConfigurableResource
from pyspark.sql import SparkSession
import os
from dotenv import load_dotenv

load_dotenv()

class SparkSessionResource(ConfigurableResource):
    app_name: str = "FilmRecommendation"
    master_url: str = "spark://spark-master:7077"

    def get_session(self) -> SparkSession:
        minio_endpoint = os.getenv("AWS_S3_ENDPOINT", "http://minio:9000")
        minio_user = os.getenv("AWS_ACCESS_KEY_ID", "admin")
        minio_password = os.getenv("AWS_SECRET_ACCESS_KEY", "admin123")

        spark = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.master_url)
            # Python version phải khớp giữa driver (Dagster) và executor (Spark workers)
            .config("spark.pyspark.python", "python3.10")
            .config("spark.pyspark.driver.python", "python3.10")
            .config("spark.driver.host", "dagster")
            # Find hadoop packages
            .config("spark.driver.extraClassPath",
                    "/opt/spark/extra-jars/hadoop-aws-3.3.4.jar:"
                    "/opt/spark/extra-jars/aws-java-sdk-bundle-1.12.262.jar")
            .config("spark.executor.extraClassPath",
                    "/opt/spark/extra-jars/hadoop-aws-3.3.4.jar:"
                    "/opt/spark/extra-jars/aws-java-sdk-bundle-1.12.262.jar")
            # Delta Lake support
            #.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            #.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            # S3A → MinIO
            .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", minio_user)
            .config("spark.hadoop.fs.s3a.secret.key", minio_password)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            # Performance
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .getOrCreate()
        )

        # Arrow optimization cho Pandas UDF (dùng trong silver.py encode BERT)
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        #Committer
        spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")
        spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored", "true")
        spark.conf.set("spark.hadoop.fs.s3a.fast.upload", "true")

        return spark