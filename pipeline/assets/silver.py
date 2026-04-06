from typing import Iterator

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from pyspark.sql.functions import col, concat_ws, arrays_zip, explode
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, DoubleType, FloatType
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf
import pandas as pd
from sentence_transformers import SentenceTransformer
from pipeline.assets.bronze import bronze_movies
from pipeline.resources.spark import SparkSessionResource
import numpy as np

_model_cache: SentenceTransformer | None = None

@asset(deps=[bronze_movies])
def silver_genres_tfidf(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    # Đọc bronze movies
    movies_df = spark.read.parquet("s3a://bronze/movies/")

    # TF-IDF cho genres: dùng HashingTF + IDF trên genre_list (array<string>)
    # HashingTF có thể xử lý array<string> trực tiếp
    hashing_tf = HashingTF(inputCol="genre_list", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="genre_tfidf")

    pipeline = Pipeline(stages=[hashing_tf, idf])
    model = pipeline.fit(movies_df)
    tfidf_df = model.transform(movies_df)

    # Lưu vector TF-IDF
    tfidf_df.select("id", "genre_tfidf").write.mode("overwrite").parquet("s3a://silver/genres_tfidf/")

    context.log.info("Silver genres TF-IDF computed and saved.")

    return MaterializeResult(
        metadata={
            "output_path": MetadataValue.text("s3a://silver/genres_tfidf/"),
        }
    )

@asset(deps=[bronze_movies])
def silver_synopsis_embeddings(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    # Đọc bronze movies
    movies_df = spark.read.parquet("s3a://bronze/movies/").select("id", "overview").coalesce(2)

    @pandas_udf(ArrayType(FloatType()))
    def encode_synopsis(overviews_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        global _model_cache
        if _model_cache is None:
            # Tắt progress bar tránh log spam trên executor
            _model_cache = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        for overviews in overviews_iter:
            texts = overviews.fillna("").tolist()
            embeddings = _model_cache.encode(
                texts,
                batch_size=16,  # Control memory per batch
                show_progress_bar=False,
                normalize_embeddings=True, # L2-norm giúp cosine sim = dot product
                convert_to_numpy=True,
            )
            yield pd.Series([row.tolist() for row in embeddings.astype(np.float32)])

    # Áp dụng UDF
    embeddings_df = movies_df.withColumn("synopsis_embedding", encode_synopsis(col("overview")))

    # Lưu embeddings
    embeddings_df.select("id", "synopsis_embedding").write.mode("overwrite").parquet("s3a://silver/synopsis_embeddings/")

    context.log.info("Silver synopsis embeddings computed and saved.")

    return MaterializeResult(
        metadata={
            "output_path": MetadataValue.text("s3a://silver/synopsis_embeddings/"),
        }
    )
