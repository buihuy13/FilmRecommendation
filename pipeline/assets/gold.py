from collections import defaultdict
import os

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Window, functions as F
from pyspark.sql.functions import col
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from pipeline.assets.bronze import bronze_ratings
from pipeline.assets.silver import silver_genres_tfidf, silver_synopsis_embeddings
from pipeline.resources.spark import SparkSessionResource


ALS_MODEL_PATH = "s3a://gold/als_model/"
QDRANT_COLLECTION = "movies"
QDRANT_BATCH_SIZE = 256
HYBRID_TOP_K = 10
HYBRID_MAX_USERS = 100
HIGH_RATING_THRESHOLD = 4.0
ALS_RANK = 20
ALS_MAX_ITER = 5
ALS_REG_PARAM = 0.1
EVAL_SAMPLE_FRACTION = 0.2


@asset(deps=[bronze_ratings])
def gold_als_model(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )
    user_means_df = ratings_df.groupBy("userId").agg(
        F.avg("rating").alias("user_mean")
    )

    # Lưu vào s3 để hybrid rcm lấy và tính mean (tránh user bias)
    user_means_df.write.mode("overwrite").parquet("s3a://gold/user_means/")

    ratings_with_mean = ratings_df.join(user_means_df, "userId", "inner").withColumn(
        "rating_norm", col("rating") - col("user_mean")
    )

    # Memory-friendly split by global timestamp quantile to avoid heavy per-user windows.
    ts_cutoff = ratings_df.approxQuantile("timestamp", [0.8], 0.01)[0]

    train_df = (
        ratings_with_mean.filter(col("timestamp") <= F.lit(ts_cutoff))
        .select("userId", "movieId", "rating_norm")
        .repartition(16, "userId")
    )

    test_df = (
        ratings_with_mean.filter(col("timestamp") > F.lit(ts_cutoff))
        .select("userId", "movieId", "rating", "user_mean", "rating_norm")
        .repartition(16, "userId")
    )

    als = ALS(
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG_PARAM,
        rank=ALS_RANK,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating_norm",
        coldStartStrategy="drop",
        nonnegative=False,
    )
    model = als.fit(train_df)

    test_for_eval = test_df.sample(False, EVAL_SAMPLE_FRACTION, seed=42)
    predictions = model.transform(test_for_eval)
    predictions = predictions.withColumn(
        "prediction_raw", col("prediction") + col("user_mean")
    )
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction_raw",
    )
    rmse = evaluator.evaluate(predictions)

    evaluated_rows = predictions.count()
    context.log.info(
        f"ALS RMSE: {rmse:.4f} on {evaluated_rows} sampled holdout rows (user-mean normalized)."
    )

    model.write().overwrite().save(ALS_MODEL_PATH)

    return MaterializeResult(
        metadata={
            "rmse": MetadataValue.float(rmse),
            "evaluated_rows": MetadataValue.int(evaluated_rows),
            "model_path": MetadataValue.text(ALS_MODEL_PATH),
            "user_means_path": MetadataValue.text("s3a://gold/user_means/"),
        }
    )


@asset(deps=[silver_genres_tfidf, silver_synopsis_embeddings])
def gold_qdrant_upsert(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    movies_df = spark.read.parquet("s3a://bronze/movies/").select(
        "id", "title", "genre_list", "release_date", "runtime", "overview"
    )
    genres_df = spark.read.parquet("s3a://silver/genres_tfidf/").select(
        "id", "genre_tfidf"
    )
    synopsis_df = spark.read.parquet("s3a://silver/synopsis_embeddings/").select(
        "id", "synopsis_embedding"
    )

    full_df = (
        movies_df.join(genres_df, "id", "inner")
        .join(synopsis_df, "id", "inner")
        .repartition(8)
    )

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    client.delete_collection(collection_name=QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "genre_tfidf": VectorParams(size=1000, distance=Distance.COSINE),
            "synopsis_embedding": VectorParams(size=384, distance=Distance.COSINE),
        },
    )

    points: list[PointStruct] = []
    count = 0

    for row in full_df.toLocalIterator():
        payload = {
            "id": row["id"],
            "title": row["title"],
            "genre_list": row["genre_list"],
            "release_date": row["release_date"],
            "runtime": row["runtime"],
            "overview": row["overview"],
        }
        vectors = {
            "genre_tfidf": row["genre_tfidf"].toArray().tolist(),
            "synopsis_embedding": row["synopsis_embedding"],
        }
        points.append(PointStruct(id=row["id"], payload=payload, vector=vectors))

        if len(points) >= QDRANT_BATCH_SIZE:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            count += len(points)
            points = []
            context.log.info(f"Upserted {count} points into Qdrant")

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        count += len(points)

    context.log.info(f"Finished Qdrant upsert: {count} points")

    return MaterializeResult(
        metadata={
            "collection": MetadataValue.text(QDRANT_COLLECTION),
            "points_upserted": MetadataValue.int(count),
        }
    )


@asset(deps=[gold_als_model, gold_qdrant_upsert])
def gold_hybrid_recommendations(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()
    model = ALSModel.load(ALS_MODEL_PATH)

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )

    active_users_df = (
        ratings_df.groupBy("userId")
        .agg(
            F.max("timestamp").alias("last_timestamp"),
            F.count("*").alias("interaction_count"),
        )
        .filter(col("interaction_count") >= 5)
        .orderBy(F.desc("last_timestamp"))
        .limit(HYBRID_MAX_USERS)
        .select("userId")
    )

    active_users = [row["userId"] for row in active_users_df.collect()]
    if not active_users:
        raise RuntimeError("No eligible users found to generate hybrid recommendations.")

    user_means_df = spark.read.parquet("s3a://gold/user_means/")

    candidates_df = (
        model.recommendForUserSubset(active_users_df, HYBRID_TOP_K)
        .select("userId", F.explode("recommendations").alias("rec"))
        .select(
            "userId",
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("collab_score_norm"),
        )
        .join(user_means_df, "userId", "left")
        .withColumn("collab_score", col("collab_score_norm") + col("user_mean"))
        .drop("collab_score_norm", "user_mean")
    )

    collab_stats = candidates_df.groupBy("userId").agg(
        F.min("collab_score").alias("collab_min"),
        F.max("collab_score").alias("collab_max"),
    )
    candidates_df = (
        candidates_df.join(collab_stats, "userId", "left")
        .withColumn(
            "collab_score_norm",
            F.when(col("collab_max") - col("collab_min") > 0,
                (col("collab_score") - col("collab_min")) / (col("collab_max") - col("collab_min"))
            ).otherwise(F.lit(1.0))
        )
        .drop("collab_min", "collab_max", "collab_score")
        .withColumnRenamed("collab_score_norm", "collab_score")
    )

    user_history_rows = (
        ratings_df.filter(col("userId").isin(active_users))
        .filter(col("rating") >= HIGH_RATING_THRESHOLD)
        .orderBy("userId", F.desc("rating"), F.desc("timestamp"))
        .select("userId", "movieId")
        .collect()
    )

    seed_movies: dict[int, int] = {}
    for row in user_history_rows:
        seed_movies.setdefault(row["userId"], row["movieId"])

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    content_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for user_id, seed_movie_id in seed_movies.items():
        retrieved = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[seed_movie_id],
            with_vectors=["synopsis_embedding"],
        )
        if not retrieved:
            continue

        query_vector = retrieved[0].vector.get("synopsis_embedding")
        if query_vector is None:
            continue

        content_results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            using="synopsis_embedding",
            limit=HYBRID_TOP_K * 2,
            with_payload=False,
        ).points

        rank = 0
        for result in content_results:
            movie_id = int(result.id)
            if movie_id == seed_movie_id:
                continue
            rank += 1
            content_scores[user_id][movie_id] = 1.0 / rank
            if rank >= HYBRID_TOP_K:
                break

    content_rows = [
        (user_id, movie_id, score)
        for user_id, movies in content_scores.items()
        for movie_id, score in movies.items()
    ]

    if content_rows:
        content_df = spark.createDataFrame(
            content_rows, ["userId", "movieId", "content_score"]
        )
    else:
        content_df = spark.createDataFrame(
            [],
            "userId int, movieId int, content_score double",
        )

    hybrid_df = (
        candidates_df.join(content_df, ["userId", "movieId"], "full_outer")
        .fillna(0.0, subset=["collab_score", "content_score"])
        .withColumn(
            "hybrid_score",
            F.lit(0.6) * col("collab_score") + F.lit(0.4) * col("content_score"),
        )
    )

    rank_window = Window.partitionBy("userId").orderBy(
        F.desc("hybrid_score"),
        F.desc("collab_score"),
        F.desc("content_score"),
        col("movieId"),
    )
    final_df = (
        hybrid_df.withColumn("rank", F.row_number().over(rank_window))
        .filter(col("rank") <= HYBRID_TOP_K)
        .drop("rank")
    )

    output_path = "s3a://gold/recommendations/hybrid/"
    final_df.write.mode("overwrite").parquet(output_path)

    users_written = final_df.select("userId").distinct().count()
    rows_written = final_df.count()
    context.log.info(
        f"Saved {rows_written} hybrid recommendations for {users_written} active users."
    )

    return MaterializeResult(
        metadata={
            "users_written": MetadataValue.int(users_written),
            "rows_written": MetadataValue.int(rows_written),
            "output_path": MetadataValue.text(output_path),
            "max_users_processed": MetadataValue.int(HYBRID_MAX_USERS),
        }
    )
