from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from pyspark.sql.functions import col, from_json, transform, regexp_replace, trim
from pyspark.sql.types import (
    ArrayType, StringType, StructType, StructField,
    IntegerType, FloatType, LongType, DoubleType
)
from pipeline.assets.ingest import upload_csv_to_minio
from pipeline.resources.spark import SparkSessionResource

_GENRE_ELEMENT_SCHEMA = ArrayType(
    StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(),  True),
    ])
)

@asset(deps=[upload_csv_to_minio])
def bronze_movies(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:

    spark = spark_resource.get_session()

    # Đọc hết dưới dạng String trước để tránh mất rows do cast thất bại.
    movies_schema = StructType([
        StructField("adult", StringType(), True),
        StructField("belongs_to_collection", StringType(), True),
        StructField("budget", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("homepage", StringType(), True),
        StructField("id", StringType(), True),
        StructField("imdb_id", StringType(), True),
        StructField("original_language", StringType(), True),
        StructField("original_title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("popularity", StringType(), True),
        StructField("poster_path", StringType(), True),
        StructField("production_companies", StringType(), True),
        StructField("production_countries", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("revenue", StringType(), True),
        StructField("runtime", StringType(), True),
        StructField("spoken_languages", StringType(), True),
        StructField("status", StringType(), True),
        StructField("tagline", StringType(), True),
        StructField("title", StringType(), True),
        StructField("video", StringType(), True),
        StructField("vote_average", StringType(), True),
        StructField("vote_count", StringType(), True),
    ])

    movies_df = spark.read.schema(movies_schema).csv(
        "s3a://landing/movies_metadata.csv",
        header=True,
        # file Kaggle có một số dòng bị lỗi quote/delimiter,
        # mode DROPMALFORMED bỏ qua thay vì làm cả batch thành null
        mode="DROPMALFORMED",
        escape='"',
        quote='"',
        multiLine=True,
    )

    context.log.info(f"Raw schema: {movies_df.dtypes}")
    movies_df.show(3, truncate=True)

    movies_df.cache()
    before_count = movies_df.count()

    clean_df = (
        movies_df
        # Lọc chỉ giữ id là số nguyên thuần tuý, loại bỏ giá trị lạ như "tt1234567"
        .filter(col("id").rlike(r"^\d+$"))
        .withColumn("id", col("id").cast(IntegerType()))
        .withColumn("runtime", col("runtime").cast(FloatType()))
        .filter(col("runtime").between(30, 300))
        .dropna(subset=["id", "title"])
        .dropDuplicates(["id"])
        .withColumn(
            "genre_list",
            transform(
                from_json(
                    regexp_replace(col("genres"), "'", '"'),
                    _GENRE_ELEMENT_SCHEMA,
                ),
                lambda g: g["name"],
            )
        )
        .filter(col("overview").isNotNull() & (trim(col("overview")) != ""))
        .withColumn("overview", trim(col("overview")))
        # Chỉ giữ các cột cần thiết cho downstream
        .select("id", "title", "genre_list", "release_date", "runtime", "overview")
    )

    after_count = clean_df.count()
    dropped = before_count - after_count
    movies_df.unpersist()

    clean_df.write.mode("overwrite").parquet("s3a://bronze/movies/")
    context.log.info(f"Bronze movies: {after_count} rows written, {dropped} rows dropped.")

    return MaterializeResult(
        metadata={
            "rows_before": MetadataValue.int(before_count),
            "rows_written": MetadataValue.int(after_count),
            "rows_dropped": MetadataValue.int(dropped),
            "output_path": MetadataValue.text("s3a://bronze/movies/"),
        }
    )

@asset(deps=[upload_csv_to_minio, bronze_movies])
def bronze_ratings(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.csv("s3a://landing/ratings.csv", header=True, inferSchema=False)
    context.log.info(f"Raw schema: {ratings_df.dtypes}")

    links_df = spark.read.csv("s3a://landing/links.csv", header=True, inferSchema=False)

    ratings_df = (
        ratings_df
        .withColumn("userId", col("userId").cast(IntegerType()))
        .withColumn("movieId", col("movieId").cast(IntegerType()))
        .withColumn("rating", col("rating").cast(FloatType()))
        .withColumn("timestamp", col("timestamp").cast(LongType()))
        # Repartition theo key trước dedup: data cùng (userId, movieId) được nhóm về cùng partition → tránh full shuffle khi dropDuplicates
        .repartition(16, "userId", "movieId")
    )

    links_df = (
        links_df
        .withColumn("movieId", col("movieId").cast(IntegerType()))
        .withColumn("tmdbId", col("tmdbId").cast(IntegerType()))
        .select("movieId", "tmdbId")
        .dropna(subset=["movieId", "tmdbId"])
        .dropDuplicates(["movieId"])
    )

    # Map MovieLens movieId -> TMDb id so ratings align with movies_metadata.id
    ratings_df = (
        ratings_df.join(links_df, "movieId", "inner")
        .withColumnRenamed("movieId", "movieId_ml")
        .withColumnRenamed("tmdbId", "movieId")
    )

    clean_df = (
        ratings_df
        .dropna(subset=["userId", "movieId", "rating"])
        .filter((col("rating") >= 0.5) & (col("rating") <= 5.0))
        .dropDuplicates(["userId", "movieId"])
    )

    # Chỉ giữ ratings có movieId tồn tại
    valid_movie_ids = (
        spark.read.parquet("s3a://bronze/movies/")
        .select(col("id").alias("movieId"))
    )
    clean_df = clean_df.join(valid_movie_ids, "movieId", "inner")

    # Cache để count() và write() dùng chung execution plan, tránh Spark recompute toàn bộ DAG 2 lần
    clean_df.cache()
    after_count = clean_df.count()

    # coalesce(16) giữ nguyên 16 partition từ bước repartition trên, tránh tạo quá nhiều file nhỏ trên MinIO
    clean_df.coalesce(16).write.mode("overwrite").parquet("s3a://bronze/ratings/")

    clean_df.unpersist()
    ratings_df.unpersist()

    context.log.info(f"Bronze ratings: {after_count} rows written")

    return MaterializeResult(
        metadata={
            "rows_written": MetadataValue.int(after_count),
            "output_path": MetadataValue.text("s3a://bronze/ratings/"),
        }
    )