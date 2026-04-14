from collections import defaultdict
import math
import os

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.functions import col
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from pipeline.assets.bronze import bronze_ratings
from pipeline.assets.silver import silver_genres_tfidf, silver_synopsis_embeddings
from pipeline.resources.spark import SparkSessionResource


ALS_MODEL_PATH = "s3a://gold/als_model/"
USER_MEANS_PATH = "s3a://gold/user_means/"
HYBRID_OUT_PATH = "s3a://gold/recommendations/hybrid/"
QDRANT_COLLECTION = "movies"
QDRANT_BATCH_SIZE = 256

ALS_RANK = 20
ALS_MAX_ITER = 5
ALS_REG_PARAM = 0.1
EVAL_SAMPLE_FRACTION = 0.2

HYBRID_TOP_K = 20
HYBRID_MAX_USERS = 1000
HYBRID_ALPHA = 0.7
HIGH_RATING_THRESHOLD = 3.0
CONTENT_SEEDS_PER_USER = 10
CONTENT_CANDIDATES_PER_SEED = HYBRID_TOP_K * 15
COLLAB_CANDIDATES_PER_USER = HYBRID_TOP_K * 15

CONTENT_GENRE_WEIGHT = 0.3
CONTENT_SYNOPSIS_WEIGHT = 0.7


# Shared helpers

def _chronological_split(df: DataFrame) -> DataFrame:
    order_w = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    stats_w = Window.partitionBy("userId")
    return (
        df.withColumn("user_event_count", F.count("*").over(stats_w))
        .withColumn("row_num", F.row_number().over(order_w))
        .withColumn(
            "eval_cutoff",
            F.greatest(
                F.lit(1),
                F.least(
                    col("user_event_count") - 1,
                    F.floor(col("user_event_count") * F.lit(0.8)).cast("int"),
                ),
            ),
        )
    )


# ALS helpers

def _build_train_test(ratings_df: DataFrame, user_means_df: DataFrame):
    ratings_with_mean = ratings_df.join(user_means_df, "userId", "inner").withColumn(
        "rating_norm", col("rating") - col("user_mean")
    )
    ranked = _chronological_split(ratings_with_mean)

    train_df = (
        ranked.filter(
            (col("user_event_count") == 1) | (col("row_num") <= col("eval_cutoff"))
        )
        .select("userId", "movieId", "rating_norm")
        .repartition(16, "userId")
    )
    test_df = (
        ranked.filter(
            (col("user_event_count") >= 2) & (col("row_num") > col("eval_cutoff"))
        )
        .select("userId", "movieId", "rating", "user_mean", "rating_norm")
        .repartition(16, "userId")
    )
    return train_df, test_df


def _evaluate_als(model, test_df: DataFrame, sample_fraction: float, context) -> tuple[float, int]:
    sample = test_df.sample(False, sample_fraction, seed=42)
    preds = model.transform(sample).withColumn(
        "prediction_raw", col("prediction") + col("user_mean")
    )
    rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction_raw"
    ).evaluate(preds)
    n = preds.count()
    context.log.info(f"ALS RMSE: {rmse:.4f} on {n} sampled holdout rows.")
    return rmse, n


# Qdrant helpers

def _recreate_qdrant_collection(client: QdrantClient) -> None:
    client.delete_collection(collection_name=QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "genre_tfidf": VectorParams(size=1000, distance=Distance.COSINE),
            "synopsis_embedding": VectorParams(size=384, distance=Distance.COSINE),
        },
    )


def _upsert_rows(client: QdrantClient, full_df: DataFrame, context) -> int:
    points: list[PointStruct] = []
    count = 0
    for row in full_df.toLocalIterator():
        points.append(
            PointStruct(
                id=row["id"],
                payload={
                    k: row[k]
                    for k in (
                        "id",
                        "title",
                        "genre_list",
                        "release_date",
                        "runtime",
                        "overview",
                    )
                },
                vector={
                    "genre_tfidf": row["genre_tfidf"].toArray().tolist(),
                    "synopsis_embedding": row["synopsis_embedding"],
                },
            )
        )
        if len(points) >= QDRANT_BATCH_SIZE:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            count += len(points)
            points = []
            context.log.info(f"Upserted {count} points into Qdrant")

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        count += len(points)
    return count


# Hybrid helpers

def _build_collab_candidates(spark: SparkSession, model, active_users_df, user_means_df, user_seen: dict[int, set[int]]) -> DataFrame:
    candidates_df = (
        model.recommendForUserSubset(active_users_df, COLLAB_CANDIDATES_PER_USER * 2)
        .select("userId", F.explode("recommendations").alias("rec"))
        .select(
            "userId",
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("_collab_raw"),
        )
        .join(user_means_df, "userId", "left")
        # Đưa về score thật (score trước đã - mean)
        .withColumn("collab_score", col("_collab_raw") + col("user_mean"))
        .drop("_collab_raw", "user_mean")
    )

    # Remove seen
    seen_rows = [(uid, mid) for uid, mids in user_seen.items() for mid in mids]
    seen_df = spark.createDataFrame(seen_rows, ["userId", "movieId"])
    candidates_df = candidates_df.join(seen_df, ["userId", "movieId"], "left_anti")

    # Z-score
    mean_w = Window.partitionBy("userId")
    mean_col = F.avg("collab_score").over(mean_w)
    std_col = F.stddev_pop("collab_score").over(mean_w)

    candidates_df = candidates_df.withColumn(
        "collab_z",
        F.when(
            std_col > 0,
            (col("collab_score") - mean_col) / std_col
        ).otherwise(F.lit(0.0))
    )

    # Optional clip
    candidates_df = candidates_df.withColumn(
        "collab_z",
        F.when(col("collab_z") > 5, 5)
         .when(col("collab_z") < -5, -5)
         .otherwise(col("collab_z"))
    )

    # Sigmoid
    candidates_df = candidates_df.withColumn(
        "collab_score",
        1 / (1 + F.exp(-col("collab_z")))
    )

    # Rank + limit
    rank_w = Window.partitionBy("userId").orderBy(F.desc("collab_score"))

    candidates_df = (
        candidates_df.withColumn("rank", F.row_number().over(rank_w))
        .filter(col("rank") <= COLLAB_CANDIDATES_PER_USER)
        .drop("rank", "collab_z")
    )

    return candidates_df

# Chưa normalize tránh user-bias
def _get_train_seed_movies(ratings_df: DataFrame, active_users: list[int]) -> dict[int, list[int]]:
    ranked = _chronological_split(ratings_df.filter(col("userId").isin(active_users)))
    history_rows = (
        ranked.filter(col("row_num") <= col("eval_cutoff"))
        # Sort theo rating cao, = rating thì sort theo timestamp -> ưu tiên rating cao và phim mới
        .orderBy("userId", F.desc("rating"), F.desc("timestamp"))
        .select("userId", "movieId")
        .collect()
    )
    # Lấy top N movieId làm seed cho mỗi userId
    seed_movies: dict[int, list[int]] = {}
    for row in history_rows:
        seeds = seed_movies.setdefault(row["userId"], [])
        if len(seeds) < CONTENT_SEEDS_PER_USER and row["movieId"] not in seeds:
            seeds.append(row["movieId"])
    return seed_movies

# Lấy các film và active_user đã xem
def _get_user_seen(ratings_df: DataFrame, user_means_df: DataFrame, active_users: list[int]) -> dict[int, set[int]]:
    ranked = _chronological_split(
        ratings_df.filter(col("userId").isin(active_users))
        .join(user_means_df, "userId", "left")
        .withColumn("rating_norm", col("rating") - col("user_mean"))
    )
    train_rows = (
        ranked.filter(
            (col("user_event_count") == 1) | (col("row_num") <= col("eval_cutoff"))
        )
        .select("userId", "movieId")
        .collect()
    )
    user_seen: dict[int, set[int]] = defaultdict(set)
    for row in train_rows:
        user_seen[row["userId"]].add(row["movieId"])
    return user_seen


def _build_content_scores(client: QdrantClient, seed_movies: dict[int, list[int]], 
                          user_seen: dict[int, set[int]]) -> list[tuple[int, int, float]]:
    content_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for user_id, seed_list in seed_movies.items():
        seen = user_seen[user_id]
        for seed_movie_id in seed_list:
            # Lấy vector của seed movie
            retrieved = client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[seed_movie_id],
                with_vectors=["synopsis_embedding", "genre_tfidf"],
            )
            if not retrieved:
                continue
            vectors = retrieved[0].vector
            synopsis_vec = vectors.get("synopsis_embedding")
            genre_vec = vectors.get("genre_tfidf")

            # Tổng hợp RR score từ cả 2 vector spaces cho seed này
            # Key: movie_id, Value: weighted RR sum (synopsis + genre)
            seed_candidate_scores: dict[int, float] = {}

            for vec, using, weight in [
                (synopsis_vec, "synopsis_embedding", CONTENT_SYNOPSIS_WEIGHT),
                (genre_vec, "genre_tfidf", CONTENT_GENRE_WEIGHT),
            ]:
                if vec is None:
                    continue
                # Lấy top CONTENT_CANDIDATES_PER_SEED phim giống với seed
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vec,
                    using=using,
                    limit=CONTENT_CANDIDATES_PER_SEED,
                    with_payload=False,
                ).points

                # Sử dụng rank-based scoring
                rank = 0
                for result in results:
                    movie_id = int(result.id)
                    if movie_id == seed_movie_id or movie_id in seen:
                        continue
                    rank += 1
                    rr_score = weight * (1.0 / math.log2(rank + 1.0))
                    seed_candidate_scores[movie_id] = (
                        seed_candidate_scores.get(movie_id, 0.0) + rr_score
                    )

            for movie_id, score in seed_candidate_scores.items():
                prev = content_scores[user_id].get(movie_id, 0.0)
                # Cần normalize lại content score sau khi cộng dồn
                content_scores[user_id][movie_id] = prev + score

    return [
        (user_id, movie_id, score)
        for user_id, movies in content_scores.items()
        for movie_id, score in movies.items()
    ]

# normalize lại content score về [0,1]
def _build_content_df(spark: SparkSession, content_rows: list[tuple]) -> DataFrame:
    if not content_rows:
        return spark.createDataFrame([], "userId int, movieId int, content_score double")

    df = spark.createDataFrame(content_rows, ["userId", "movieId", "content_score"])

    # Window theo user
    mean_w = Window.partitionBy("userId")

    mean_col = F.avg("content_score").over(mean_w)
    std_col = F.stddev_pop("content_score").over(mean_w)

    # Z-score
    df = df.withColumn(
        "content_z",
        F.when(
            std_col > 0,
            (col("content_score") - mean_col) / std_col
        ).otherwise(F.lit(0.0))
    )

    # Optional clip
    df = df.withColumn(
        "content_z",
        F.when(col("content_z") > 5, 5)
         .when(col("content_z") < -5, -5)
         .otherwise(col("content_z"))
    )

    # Sigmoid → [0,1]
    df = df.withColumn(
        "content_score",
        1 / (1 + F.exp(-col("content_z")))
    )

    return df.drop("content_z")


def _blend_and_rank(candidates_df: DataFrame, content_df: DataFrame) -> DataFrame:
    all_candidates = candidates_df.select("userId", "movieId").union(content_df.select("userId", "movieId")).dropDuplicates()
    hybrid_df = (
        #candidates_df
        #.join(content_df, ["userId", "movieId"], "left")
        #.union(content_df.select("userId", "movieId"))
        all_candidates
        .join(candidates_df, ["userId", "movieId"], "left")
        .join(content_df, ["userId", "movieId"], "left")
        .fillna(0.0)
        #.fillna(0.0, subset=["content_score"])
        .withColumn(
            "hybrid_score",
            F.lit(HYBRID_ALPHA) * col("collab_score")
            + F.lit(1.0 - HYBRID_ALPHA) * col("content_score"),
        )
    )
    rank_w = Window.partitionBy("userId").orderBy(
        F.desc("hybrid_score"),
        F.desc("collab_score"),
        F.desc("content_score"),
        col("movieId"),
    )
    return (
        hybrid_df.withColumn("rank", F.row_number().over(rank_w))
        .filter(col("rank") <= HYBRID_TOP_K)
        .drop("rank")
    )


# Dagster assets

@asset(deps=[bronze_ratings])
def gold_als_model(
    context: AssetExecutionContext, spark_resource: SparkSessionResource
) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )
    user_means_df = ratings_df.groupBy("userId").agg(F.avg("rating").alias("user_mean"))
    user_means_df.write.mode("overwrite").parquet(USER_MEANS_PATH)

    train_df, test_df = _build_train_test(ratings_df, user_means_df)

    model = ALS(
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG_PARAM,
        rank=ALS_RANK,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating_norm",
        coldStartStrategy="drop",
        nonnegative=False,
    ).fit(train_df)

    rmse, evaluated_rows = _evaluate_als(model, test_df, EVAL_SAMPLE_FRACTION, context)
    model.write().overwrite().save(ALS_MODEL_PATH)

    return MaterializeResult(
        metadata={
            "rmse": MetadataValue.float(rmse),
            "evaluated_rows": MetadataValue.int(evaluated_rows),
            "model_path": MetadataValue.text(ALS_MODEL_PATH),
            "user_means_path": MetadataValue.text(USER_MEANS_PATH),
        }
    )


@asset(deps=[silver_genres_tfidf, silver_synopsis_embeddings])
def gold_qdrant_upsert(
    context: AssetExecutionContext, spark_resource: SparkSessionResource
) -> MaterializeResult:
    spark = spark_resource.get_session()

    full_df = (
        spark.read.parquet("s3a://bronze/movies/")
        .select("id", "title", "genre_list", "release_date", "runtime", "overview")
        .join(
            spark.read.parquet("s3a://silver/genres_tfidf/").select(
                "id", "genre_tfidf"
            ),
            "id",
            "inner",
        )
        .join(
            spark.read.parquet("s3a://silver/synopsis_embeddings/").select(
                "id", "synopsis_embedding"
            ),
            "id",
            "inner",
        )
        .repartition(8)
    )

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    _recreate_qdrant_collection(client)
    count = _upsert_rows(client, full_df, context)
    context.log.info(f"Finished Qdrant upsert: {count} points")

    return MaterializeResult(
        metadata={
            "collection": MetadataValue.text(QDRANT_COLLECTION),
            "points_upserted": MetadataValue.int(count),
        }
    )


@asset(deps=[gold_als_model, gold_qdrant_upsert])
def gold_hybrid_recommendations(
    context: AssetExecutionContext, spark_resource: SparkSessionResource
) -> MaterializeResult:
    spark = spark_resource.get_session()
    model = ALSModel.load(ALS_MODEL_PATH)
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )
    user_means_df = spark.read.parquet(USER_MEANS_PATH)

    active_users_df = (
        ratings_df.groupBy("userId")
        .agg(
            F.max("timestamp").alias("last_timestamp"),
            F.count("*").alias("interaction_count"),
        )
        .filter(col("interaction_count") >= 5)
        .orderBy(F.rand(seed=42))
        #.orderBy(F.desc("last_timestamp"))
        .limit(HYBRID_MAX_USERS)
        .select("userId")
    )
    active_users = [row["userId"] for row in active_users_df.collect()]
    if not active_users:
        raise RuntimeError(
            "No eligible users found to generate hybrid recommendations."
        )

    seed_movies = _get_train_seed_movies(ratings_df, active_users)
    user_seen = _get_user_seen(ratings_df, user_means_df, active_users)
    candidates_df = _build_collab_candidates(spark, model, active_users_df, user_means_df, user_seen)
    content_rows = _build_content_scores(client, seed_movies, user_seen)
    content_df = _build_content_df(spark, content_rows)
    final_df = _blend_and_rank(candidates_df, content_df)

    final_df.write.mode("overwrite").parquet(HYBRID_OUT_PATH)

    users_written = final_df.select("userId").distinct().count()
    rows_written = final_df.count()
    context.log.info(
        f"Saved {rows_written} hybrid recommendations for {users_written} active users."
    )

    return MaterializeResult(
        metadata={
            "users_written": MetadataValue.int(users_written),
            "rows_written": MetadataValue.int(rows_written),
            "output_path": MetadataValue.text(HYBRID_OUT_PATH),
            "max_users_processed": MetadataValue.int(HYBRID_MAX_USERS),
        }
    )