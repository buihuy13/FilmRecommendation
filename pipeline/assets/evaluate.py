from multiprocessing import context

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, Window, functions as F
from pyspark.sql.functions import col

from pipeline.assets.gold import (
    HYBRID_ALPHA,
    gold_als_model,
    gold_hybrid_recommendations,
)
from pipeline.resources.spark import SparkSessionResource


ALS_MODEL_PATH = "s3a://gold/als_model/"
USER_MEANS_PATH = "s3a://gold/user_means/"
HYBRID_RECS_PATH = "s3a://gold/recommendations/hybrid/"

EVAL_SAMPLE_FRACTION = 0.2
RECS_TOP_K = 10
RECS_USER_SAMPLE = 1000
HYBRID_RELEVANT_THRESHOLD = 3.5


def _chronological_test_df(ratings_df: DataFrame) -> DataFrame:
    order_w = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    stats_w = Window.partitionBy("userId")
    ranked = (
        ratings_df.withColumn("user_event_count", F.count("*").over(stats_w))
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
    return ranked.filter(
        (col("user_event_count") >= 2) & (col("row_num") > col("eval_cutoff"))
    ).select("userId", "movieId", "rating", "timestamp")


@asset(deps=[gold_als_model])
def evaluate_als(
    context: AssetExecutionContext, spark_resource: SparkSessionResource
) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
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
        ratings_df.select("userId").distinct().orderBy(F.rand()).limit(RECS_USER_SAMPLE)
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
def evaluate_hybrid(
    context: AssetExecutionContext, spark_resource: SparkSessionResource
) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )

    full_test_df = _chronological_test_df(ratings_df).cache()

    context.log.info(f"Full test rows: {full_test_df.count()}")

    hybrid_df = spark.read.parquet(HYBRID_RECS_PATH).select(
        "userId", "movieId", "hybrid_score"
    )

    overlap_count = hybrid_df.join(
        full_test_df.select("userId", "movieId"),
        ["userId", "movieId"],
        "inner"
    ).count()
    context.log.info(f"Overlap giữa recommendations và test set: {overlap_count}")
    eval_users_df = hybrid_df.select("userId").distinct()

    # seen = toàn bộ ratings của eval_users KHÔNG nằm trong test set của họ
    # (= phần train của eval_users)
    # seen_df = (
    #     ratings_df.join(eval_users_df, "userId", "inner")   # chỉ lấy eval_users
    #     .join(full_test_df, ["userId", "movieId"], "left_anti")  # bỏ test rows
    #     .select("userId", "movieId")
    #     .distinct()
    # )

    # context.log.info(f"Seen pairs: {seen_df.count()}")


    # test_df dùng để tính relevant: chỉ cần test rows của eval_users
    test_df = full_test_df.join(eval_users_df, "userId", "inner")

    relevant_df = (
        test_df.filter(col("rating") >= HYBRID_RELEVANT_THRESHOLD)
        .select("userId", col("movieId").alias("rel_movieId"))
        .dropDuplicates(["userId", "rel_movieId"])
    )

    context.log.info(f"Total relevant pairs: {relevant_df.count()}")
    context.log.info(f"Avg relevant per user:")
    relevant_df.groupBy("userId").count().agg(F.avg("count")).show()

    # Lọc ra những recommendations chưa được user xem trong train
    # candidates_df = hybrid_df.join(seen_df, ["userId", "movieId"], "left_anti")

    candidates_df = hybrid_df

    cand_count = candidates_df.count()
    cand_users = candidates_df.select("userId").distinct().count()
    context.log.info(f"Candidates sau lọc seen: {cand_count} rows, {cand_users} users")

    rank_w = Window.partitionBy("userId").orderBy(
        F.desc("hybrid_score"),
        col("movieId"),  # tiebreaker deterministc
    )

    k_recs_df = (
        candidates_df.withColumn("rank", F.row_number().over(rank_w))
        .filter(col("rank") <= RECS_TOP_K)
        .select("userId", "movieId", "rank")
    )

    krecs_count = k_recs_df.count()
    krecs_users = k_recs_df.select("userId").distinct().count()
    context.log.info(f"k_recs (top-{RECS_TOP_K}): {krecs_count} rows, {krecs_users} users")

    hits_df = k_recs_df.join(
        relevant_df,
        (k_recs_df.userId == relevant_df.userId)
        & (k_recs_df.movieId == relevant_df.rel_movieId),
        "inner",
    ).select(k_recs_df.userId, k_recs_df.movieId)

    context.log.info(f"Total hits: {hits_df.count()}")


    hit_counts = hits_df.groupBy("userId").agg(F.count("*").alias("hit_count"))
    rel_counts = relevant_df.groupBy("userId").agg(F.count("*").alias("rel_count"))
    rec_counts = k_recs_df.groupBy("userId").agg(F.count("*").alias("rec_count"))

    metrics_df = (
        eval_users_df
        .join(rec_counts, "userId", "left")
        .join(rel_counts, "userId", "left")
        .join(hit_counts, "userId", "left")
        .fillna(0, subset=["hit_count", "rel_count", "rec_count"])
        .withColumn(
            "precision_at_k",
            F.when(col("rec_count") > 0, col("hit_count") / col("rec_count")).otherwise(0.0),
        )
        .withColumn(
            "recall_at_k",
            F.when(col("rel_count") > 0, col("hit_count") / col("rel_count")).otherwise(0.0),
        )
        .withColumn(
            "hit_rate",
            F.when(col("hit_count") > 0, 1.0).otherwise(0.0),
        )
    )

    ranked_hits = (
        k_recs_df.alias("recs")
        .join(
            relevant_df.alias("rel"),
            (col("recs.userId") == col("rel.userId"))
            & (col("recs.movieId") == col("rel.rel_movieId")),
            "left",
        )
        .select(
            col("recs.userId"),
            col("recs.rank"),
            F.when(col("rel.rel_movieId").isNotNull(), 1.0).otherwise(0.0).alias("is_relevant"),
        )
    )

    dcg_df = (
        ranked_hits.withColumn(
            "dcg_contrib",
            F.when(
                col("is_relevant") > 0,
                1.0 / F.log2(col("rank") + 1.0),
            ).otherwise(0.0),
        )
        .groupBy("userId")
        .agg(F.sum("dcg_contrib").alias("dcg"))
    )

    ideal_df = rel_counts.withColumn(
        "ideal_dcg",
        F.expr(
            f"aggregate(sequence(1, least(rel_count, {RECS_TOP_K})), "
            f"0.0D, (acc, x) -> acc + 1.0D / log2(x + 1))"
        ),
    )

    ndcg_df = (
        dcg_df.join(ideal_df, "userId", "left")
        .withColumn(
            "ndcg_at_k",
            F.when(col("ideal_dcg") > 0, col("dcg") / col("ideal_dcg")).otherwise(0.0),
        )
        .select("userId", "ndcg_at_k")
    )

    row = (
        metrics_df.join(ndcg_df, "userId", "left")
        .fillna(0.0, subset=["ndcg_at_k"])
        .agg(
            F.avg("precision_at_k").alias("precision_at_k"),
            F.avg("recall_at_k").alias("recall_at_k"),
            F.avg("hit_rate").alias("hit_rate"),
            F.avg("ndcg_at_k").alias("ndcg_at_k"),
            F.count("*").alias("users_evaluated"),
        )
        .collect()[0]
    )

    full_test_df.unpersist()

    precision_at_k = float(row["precision_at_k"] or 0.0)
    recall_at_k = float(row["recall_at_k"] or 0.0)
    hit_rate = float(row["hit_rate"] or 0.0)
    ndcg_at_k = float(row["ndcg_at_k"] or 0.0)
    users_evaluated = int(row["users_evaluated"] or 0)

    users_with_relevant = relevant_df.select("userId").distinct().count()
    context.log.info(f"Users có relevant items trong test: {users_with_relevant} / {users_evaluated}")

    context.log.info(
        "Hybrid eval (alpha=%.2f): users=%s precision@%s=%.4f "
        "recall@%s=%.4f hit_rate=%.4f ndcg@%s=%.4f",
        HYBRID_ALPHA,
        users_evaluated,
        RECS_TOP_K,
        precision_at_k,
        RECS_TOP_K,
        recall_at_k,
        hit_rate,
        RECS_TOP_K,
        ndcg_at_k,
    )

    return MaterializeResult(
        metadata={
            "users_evaluated": MetadataValue.int(users_evaluated),
            "precision_at_k": MetadataValue.float(precision_at_k),
            "recall_at_k": MetadataValue.float(recall_at_k),
            "hit_rate": MetadataValue.float(hit_rate),
            "ndcg_at_k": MetadataValue.float(ndcg_at_k),
            "alpha_used": MetadataValue.float(HYBRID_ALPHA),
            "top_k": MetadataValue.int(RECS_TOP_K),
        }
    )