from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, Window, functions as F
from pyspark.sql.functions import col

from pipeline.assets.gold import (
    ALS_MODEL_PATH,
    HYBRID_OUT_PATH,
    HYBRID_TOP_K,
    HIGH_RATING_THRESHOLD,
    USER_MEANS_PATH,
    _chronological_split,
    gold_als_model,
    gold_hybrid_recommendations,
)
from pipeline.resources.spark import SparkSessionResource


BRONZE_RATINGS_PATH = "s3a://bronze/ratings/"
EVAL_SAMPLE_FRACTION = 0.2
RECS_TOP_K = HYBRID_TOP_K
RECS_USER_SAMPLE = 1000


def _chronological_test_df(ratings_df: DataFrame) -> DataFrame:
    ranked = _chronological_split(ratings_df)
    return ranked.filter(
        (col("user_event_count") >= 2) & (col("row_num") > col("eval_cutoff"))
    ).select("userId", "movieId", "rating", "timestamp")


@asset(deps=[gold_als_model])
def evaluate_als(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet(BRONZE_RATINGS_PATH).select(
        "userId", "movieId", "rating", "timestamp"
    )

    num_users = ratings_df.select("userId").distinct().count()
    num_items = ratings_df.select("movieId").distinct().count()
    num_ratings = ratings_df.count()
    total_possible = num_users * num_items if num_users and num_items else 0
    sparsity = 1.0 - (num_ratings / total_possible) if total_possible else 1.0

    model = ALSModel.load(ALS_MODEL_PATH)
    user_means_df = spark.read.parquet(USER_MEANS_PATH)

    test_df = _chronological_test_df(ratings_df)
    sample = (
        test_df.join(user_means_df, "userId", "left")
        .withColumn("rating_norm", col("rating") - col("user_mean"))
        .sample(False, EVAL_SAMPLE_FRACTION, seed=42)
    )
    preds = model.transform(sample).withColumn(
        "prediction_raw", col("prediction") + col("user_mean")
    )

    rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction_raw"
    ).evaluate(preds)
    mae = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction_raw"
    ).evaluate(preds)

    eval_rows = preds.count()
    test_rows = sample.count()
    prediction_coverage = (eval_rows / test_rows) if test_rows else 0.0

    sample_users_df = (
        ratings_df.select("userId").distinct().orderBy(F.rand(seed=42)).limit(RECS_USER_SAMPLE)
    )
    users_with_recs = (
        model.recommendForUserSubset(sample_users_df, RECS_TOP_K)
        .select("userId")
        .distinct()
        .count()
    )

    context.log.info(
        "ALS eval: users=%s items=%s ratings=%s sparsity=%.6f "
        "rmse=%.4f mae=%.4f coverage=%.4f users_with_recs=%s",
        num_users,
        num_items,
        num_ratings,
        sparsity,
        rmse,
        mae,
        prediction_coverage,
        users_with_recs,
    )

    return MaterializeResult(
        metadata={
            "num_users": MetadataValue.int(num_users),
            "num_items": MetadataValue.int(num_items),
            "num_ratings": MetadataValue.int(num_ratings),
            "sparsity": MetadataValue.float(sparsity),
            "rmse": MetadataValue.float(rmse),
            "mae": MetadataValue.float(mae),
            "prediction_coverage": MetadataValue.float(prediction_coverage),
            "users_with_recs": MetadataValue.int(users_with_recs),
            "recs_user_sample": MetadataValue.int(RECS_USER_SAMPLE),
            "recs_top_k": MetadataValue.int(RECS_TOP_K),
        }
    )


@asset(deps=[gold_hybrid_recommendations])
def evaluate_hybrid_sampled(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    import random

    spark = spark_resource.get_session()
    NUM_NEGATIVES = 99  # 1 relevant + 99 random negatives
    SAMPLE_SEED = 42

    ratings_df = spark.read.parquet(BRONZE_RATINGS_PATH).select(
        "userId", "movieId", "rating", "timestamp"
    )
    hybrid_df = spark.read.parquet(HYBRID_OUT_PATH).select(
        "userId", "movieId", "hybrid_score"
    )

    hybrid_users_df = hybrid_df.select("userId").distinct().cache()
    if hybrid_users_df.rdd.isEmpty():
        raise RuntimeError("No hybrid recommendations found to evaluate.")

    # Positive item phải đến từ phần holdout thật của gold, không lấy từ toàn bộ lịch sử.
    full_test_df = (
        _chronological_test_df(ratings_df)
        .join(hybrid_users_df, "userId", "inner")
        .filter(col("rating") >= HIGH_RATING_THRESHOLD)  # chỉ lấy item user thích
        .cache()
    )
    test_rank_w = Window.partitionBy("userId").orderBy(
        F.desc("rating"),
        F.desc("timestamp"),
        F.desc("movieId"),
    )
    test_item_df = (
        full_test_df.withColumn("rn", F.row_number().over(test_rank_w))
        .filter(col("rn") <= 3)
        .select("userId", col("movieId").alias("test_movieId"))
        .cache()
    )

    eval_users_df = test_item_df.select("userId").cache()
    context.log.info(f"Users có test item: {eval_users_df.count()}")

    # Retrieval coverage
    users_with_test = test_item_df.count()
    positive_retrieved = (
        test_item_df
        .join(hybrid_df,
              (test_item_df.userId == hybrid_df.userId) &
              (test_item_df.test_movieId == hybrid_df.movieId),
              "inner")
        .count()
    )
    context.log.info(
        f"Positive retrieved bởi model: {positive_retrieved}/{users_with_test} "
        f"= {positive_retrieved/users_with_test:.4f}"
    )

    # Negative pool loại toàn bộ item user từng rate, để tránh lấy nhầm positive khác làm negative.
    all_rated_rows = (
        ratings_df.join(eval_users_df, "userId", "inner")
        .select("userId", "movieId")
        .collect()
    )
    user_rated: dict[int, set[int]] = {}
    for row in all_rated_rows:
        user_rated.setdefault(row["userId"], set()).add(row["movieId"])

    all_movies = [row["movieId"] for row in ratings_df.select("movieId").distinct().collect()]
    all_movies_set = set(all_movies)

    # Build sampled candidate pool: 1 positive + 99 negatives per user
    rng = random.Random(SAMPLE_SEED)
    sample_rows = []
    test_item_map = {
        row["userId"]: row["test_movieId"]
        for row in test_item_df.collect()
    }

    for user_id, test_movie_id in test_item_map.items():
        rated = user_rated.get(user_id, set())
        neg_pool = list(all_movies_set - rated - {test_movie_id})
        if len(neg_pool) < NUM_NEGATIVES:
            continue
        negatives = rng.sample(neg_pool, NUM_NEGATIVES)
        sample_rows.append((user_id, test_movie_id, 1))
        for neg_id in negatives:
            sample_rows.append((user_id, neg_id, 0))

    candidate_pool_df = spark.createDataFrame(
        sample_rows, ["userId", "movieId", "is_positive"]
    ).cache()
    context.log.info(f"Candidate pool: {candidate_pool_df.count()} rows")

    # Vì gold chỉ lưu top-K cuối cùng, item không có score được coi là "không được retrieve".
    scored_df = (
        candidate_pool_df.alias("cand")
        .join(hybrid_df, ["userId", "movieId"], "left")
        .withColumn("has_score", F.when(col("hybrid_score").isNotNull(), 1).otherwise(0))
        .cache()
    )

    scored_rank_w = Window.partitionBy("userId").orderBy(
        F.desc(F.coalesce(col("hybrid_score"), F.lit(-1.0))),
        col("movieId"),
    )

    scored_ranked_df = (
        scored_df
        .withColumn("rank_in_pool", F.row_number().over(scored_rank_w))
        .select("userId", "movieId", "rank_in_pool")
    )

    positive_rank_df = (
        test_item_df.join(
            scored_ranked_df,
            (test_item_df.userId == scored_ranked_df.userId)
            & (test_item_df.test_movieId == scored_ranked_df.movieId),
            "left",
        )
        .select(
            test_item_df.userId,
            F.when(col("rank_in_pool").isNotNull(), col("rank_in_pool"))
            .otherwise(F.lit(NUM_NEGATIVES + 1))
            .alias("rank_in_pool"),
        )
        .cache()
    )

    agg_row = (
        positive_rank_df
        .withColumn(
            "hit_at_1",
            F.when(col("rank_in_pool") <= 1, 1.0).otherwise(0.0),
        )
        .withColumn(
            "hit_at_5",
            F.when(col("rank_in_pool") <= 5, 1.0).otherwise(0.0),
        )
        .withColumn(
            "hit_at_10",
            F.when(col("rank_in_pool") <= 10, 1.0).otherwise(0.0),
        )
        .withColumn(
            "hit_at_20",
            F.when(col("rank_in_pool") <= RECS_TOP_K, 1.0).otherwise(0.0),
        )
        .withColumn(
            "ndcg_at_k",
            F.when(
                col("rank_in_pool") <= RECS_TOP_K,
                1.0 / F.log2(col("rank_in_pool") + 1.0),
            ).otherwise(0.0),
        )
        .agg(
            F.avg("hit_at_1").alias("hit_at_1"),
            F.avg("hit_at_5").alias("hit_at_5"),
            F.avg("hit_at_10").alias("hit_at_10"),
            F.avg("hit_at_20").alias("hit_at_20"),
            F.avg("ndcg_at_k").alias("ndcg_at_k"),
            F.count("*").alias("users_evaluated"),
        )
        .collect()[0]
    )

    positive_scored_rate = (
        test_item_df
        .join(
            hybrid_df,
            (test_item_df.userId == hybrid_df.userId) &
            (test_item_df.test_movieId == hybrid_df.movieId),
            "left_semi"
        )
        .count() / users_with_test
    )
    hit_at_1 = float(agg_row["hit_at_1"] or 0.0)
    hit_at_5 = float(agg_row["hit_at_5"] or 0.0)
    hit_at_10 = float(agg_row["hit_at_10"] or 0.0)
    hit_at_20 = float(agg_row["hit_at_20"] or 0.0)
    ndcg_at_k = float(agg_row["ndcg_at_k"] or 0.0)
    users_evaluated = int(agg_row["users_evaluated"] or 0)

    context.log.info(
        "Sampled hybrid eval (%s negatives): users=%s positive_scored_rate=%.4f "
        "hit@1=%.4f hit@5=%.4f hit@10=%.4f hit@%s=%.4f ndcg@%s=%.4f",
        NUM_NEGATIVES,
        users_evaluated,
        positive_scored_rate,
        hit_at_1,
        hit_at_5,
        hit_at_10,
        RECS_TOP_K,
        hit_at_20,
        RECS_TOP_K,
        ndcg_at_k,
    )

    hybrid_users_df.unpersist()
    full_test_df.unpersist()
    test_item_df.unpersist()
    eval_users_df.unpersist()
    candidate_pool_df.unpersist()
    scored_df.unpersist()
    positive_rank_df.unpersist()

    return MaterializeResult(
        metadata={
            "users_evaluated": MetadataValue.int(users_evaluated),
            "num_negatives": MetadataValue.int(NUM_NEGATIVES),
            "positive_scored_rate": MetadataValue.float(positive_scored_rate),
            "hit_at_1": MetadataValue.float(hit_at_1),
            "hit_at_5": MetadataValue.float(hit_at_5),
            "hit_at_10": MetadataValue.float(hit_at_10),
            "hit_at_20": MetadataValue.float(hit_at_20),
            "ndcg_at_k": MetadataValue.float(ndcg_at_k),
            "top_k": MetadataValue.int(RECS_TOP_K),
        }
    )
