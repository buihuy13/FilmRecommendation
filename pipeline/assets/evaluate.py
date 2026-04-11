from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Window, functions as F
from pyspark.sql.functions import col

from pipeline.assets.gold import gold_als_model, gold_hybrid_recommendations
from pipeline.resources.spark import SparkSessionResource


ALS_MODEL_PATH = "s3a://gold/als_model/"
USER_MEANS_PATH = "s3a://gold/user_means/"
EVAL_SAMPLE_FRACTION = 0.2
RECS_TOP_K = 10
RECS_USER_SAMPLE = 1000
HYBRID_RELEVANT_THRESHOLD = 3.5
HYBRID_RECS_PATH = "s3a://gold/recommendations/hybrid/"
ALPHA_CANDIDATES = [0.2, 0.4, 0.6, 0.8]


@asset(deps=[gold_als_model])
def evaluate_als(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )

    # Basic dataset stats
    num_users = ratings_df.select("userId").distinct().count()
    num_items = ratings_df.select("movieId").distinct().count()
    num_ratings = ratings_df.count()
    total_possible = num_users * num_items if num_users and num_items else 0
    sparsity = 1.0 - (num_ratings / total_possible) if total_possible else 1.0

    # Load model + user means for de-normalization
    model = ALSModel.load(ALS_MODEL_PATH)
    user_means_df = spark.read.parquet(USER_MEANS_PATH)

    # Per-user chronological holdout (align with gold layer)
    order_window = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    stats_window = Window.partitionBy("userId")
    ranked_df = (
        ratings_df.withColumn("user_event_count", F.count("*").over(stats_window))
        .withColumn("row_num", F.row_number().over(order_window))
        .withColumn(
            "eval_cutoff",
            F.greatest(
                F.lit(1),
                F.least(
                    F.col("user_event_count") - 1,
                    F.floor(F.col("user_event_count") * F.lit(0.8)).cast("int"),
                ),
            ),
        )
    )

    test_df = ranked_df.filter(
        (col("user_event_count") >= 2) & (col("row_num") > col("eval_cutoff"))
    ).select("userId", "movieId", "rating", "timestamp")

    test_with_mean = (
        test_df.join(user_means_df, "userId", "left")
        .withColumn("rating_norm", col("rating") - col("user_mean"))
    )

    test_for_eval = test_with_mean.sample(False, EVAL_SAMPLE_FRACTION, seed=42)
    predictions = model.transform(test_for_eval)
    predictions = predictions.withColumn(
        "prediction_raw", col("prediction") + col("user_mean")
    )

    rmse = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction_raw",
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction_raw",
    ).evaluate(predictions)

    eval_rows = predictions.count()
    test_rows = test_for_eval.count()
    prediction_coverage = (eval_rows / test_rows) if test_rows else 0.0

    # Coverage of recommendForUserSubset on a user sample
    sample_users_df = (
        ratings_df.select("userId").distinct().orderBy(F.rand()).limit(RECS_USER_SAMPLE)
    )
    recs_df = model.recommendForUserSubset(sample_users_df, RECS_TOP_K)
    users_with_recs = recs_df.select("userId").distinct().count()

    context.log.info(
        "ALS eval: users=%s items=%s ratings=%s sparsity=%.6f rmse=%.4f mae=%.4f "
        "coverage=%.4f users_with_recs=%s",
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
def evaluate_hybrid(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    ratings_df = spark.read.parquet("s3a://bronze/ratings/").select(
        "userId", "movieId", "rating", "timestamp"
    )
    hybrid_df = spark.read.parquet(HYBRID_RECS_PATH).select(
        "userId", "movieId", "collab_score", "content_score"
    )

    # Align evaluation users with hybrid output
    eval_users_df = hybrid_df.select("userId").distinct()

    # Per-user chronological test set (same as evaluate_als)
    order_window = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    stats_window = Window.partitionBy("userId")
    ranked_df = (
        ratings_df.withColumn("user_event_count", F.count("*").over(stats_window))
        .withColumn("row_num", F.row_number().over(order_window))
        .withColumn(
            "eval_cutoff",
            F.greatest(
                F.lit(1),
                F.least(
                    F.col("user_event_count") - 1,
                    F.floor(F.col("user_event_count") * F.lit(0.8)).cast("int"),
                ),
            ),
        )
    )
    test_df = ranked_df.filter(
        (col("user_event_count") >= 2) & (col("row_num") > col("eval_cutoff"))
    ).select("userId", "movieId", "rating", "timestamp")

    # Define relevant items by high rating in test
    relevant_df = (
        test_df.filter(col("rating") >= F.lit(HYBRID_RELEVANT_THRESHOLD))
        .select("userId", col("movieId").alias("rel_movieId"))
        .join(eval_users_df, "userId", "inner")
        .dropDuplicates(["userId", "rel_movieId"])
    )

    k_recs_df = hybrid_df.dropDuplicates(["userId", "movieId"])

    hits_df = k_recs_df.join(
        relevant_df,
        (k_recs_df.userId == relevant_df.userId)
        & (k_recs_df.movieId == relevant_df.rel_movieId),
        "inner",
    ).select(k_recs_df.userId, k_recs_df.movieId)

    hit_counts = hits_df.groupBy("userId").agg(
        F.count("*").alias("hit_count")
    )
    rel_counts = relevant_df.groupBy("userId").agg(
        F.count("*").alias("rel_count")
    )
    rec_counts = k_recs_df.groupBy("userId").agg(
        F.count("*").alias("rec_count")
    )

    metrics_df = (
        rec_counts.join(rel_counts, "userId", "left")
        .join(hit_counts, "userId", "left")
        .fillna(0, subset=["hit_count", "rel_count"])
        .withColumn(
            "precision_at_k",
            F.when(col("rec_count") > 0, col("hit_count") / col("rec_count")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "recall_at_k",
            F.when(col("rel_count") > 0, col("hit_count") / col("rel_count")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "hit_rate",
            F.when(col("hit_count") > 0, F.lit(1.0)).otherwise(F.lit(0.0))
        )
    )

    # Evaluate only users who actually have relevant items in the test set.
    metrics_df = metrics_df.filter(col("rel_count") > 0)

    best_alpha = None
    best_ndcg = -1.0
    best_metrics = None

    for alpha in ALPHA_CANDIDATES:
        scored_df = k_recs_df.withColumn(
            "hybrid_score",
            F.lit(alpha) * col("collab_score") + F.lit(1.0 - alpha) * col("content_score"),
        )

        rank_window = Window.partitionBy("userId").orderBy(
            F.desc("hybrid_score"), col("movieId")
        )
        ranked_recs = scored_df.withColumn("rank", F.row_number().over(rank_window))

        recs_alias = ranked_recs.alias("recs")
        rel_alias = relevant_df.alias("rel")
        ranked_hits = (
            recs_alias.join(
                rel_alias,
                (col("recs.userId") == col("rel.userId"))
                & (col("recs.movieId") == col("rel.rel_movieId")),
                "left",
            )
            .select(
                col("recs.userId").alias("userId"),
                col("recs.movieId").alias("movieId"),
                col("recs.rank").alias("rank"),
                col("rel.rel_movieId").alias("rel_movieId"),
            )
            .withColumn(
                "is_relevant",
                F.when(col("rel_movieId").isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
        )

        dcg_df = ranked_hits.withColumn(
            "dcg_contrib",
            F.when(
                col("is_relevant") > 0,
                F.lit(1.0) / F.log2(col("rank") + F.lit(1.0)),
            ).otherwise(F.lit(0.0))
        ).groupBy("userId").agg(F.sum("dcg_contrib").alias("dcg"))

        ideal_df = rel_counts.withColumn(
            "ideal_dcg",
            F.expr(
                "aggregate(sequence(1, least(rel_count, {})), 0.0D, (acc, x) -> acc + 1.0D / log2(x + 1))".format(
                    RECS_TOP_K
                )
            ),
        )

        ndcg_df = (
            dcg_df.join(ideal_df, "userId", "left")
            .withColumn(
                "ndcg_at_k",
                F.when(col("ideal_dcg") > 0, col("dcg") / col("ideal_dcg")).otherwise(F.lit(0.0))
            )
        )

        final_metrics = (
            metrics_df.join(ndcg_df.select("userId", "ndcg_at_k"), "userId", "left")
            .agg(
                F.avg("precision_at_k").alias("precision_at_k"),
                F.avg("recall_at_k").alias("recall_at_k"),
                F.avg("hit_rate").alias("hit_rate"),
                F.avg("ndcg_at_k").alias("ndcg_at_k"),
                F.count("*").alias("users_evaluated"),
            )
            .collect()[0]
        )

        precision_at_k = float(final_metrics["precision_at_k"] or 0.0)
        recall_at_k = float(final_metrics["recall_at_k"] or 0.0)
        hit_rate = float(final_metrics["hit_rate"] or 0.0)
        ndcg_at_k = float(final_metrics["ndcg_at_k"] or 0.0)
        users_evaluated = int(final_metrics["users_evaluated"] or 0)

        context.log.info(
            "Hybrid eval (alpha=%.2f): users=%s precision@%s=%.4f recall@%s=%.4f hit_rate=%.4f ndcg@%s=%.4f",
            alpha,
            users_evaluated,
            RECS_TOP_K,
            precision_at_k,
            RECS_TOP_K,
            recall_at_k,
            hit_rate,
            RECS_TOP_K,
            ndcg_at_k,
        )

        if ndcg_at_k > best_ndcg:
            best_ndcg = ndcg_at_k
            best_alpha = alpha
            best_metrics = (precision_at_k, recall_at_k, hit_rate, ndcg_at_k, users_evaluated)

    precision_at_k, recall_at_k, hit_rate, ndcg_at_k, users_evaluated = best_metrics or (0.0, 0.0, 0.0, 0.0, 0)

    return MaterializeResult(
        metadata={
            "users_evaluated": MetadataValue.int(users_evaluated),
            "precision_at_k": MetadataValue.float(precision_at_k),
            "recall_at_k": MetadataValue.float(recall_at_k),
            "hit_rate": MetadataValue.float(hit_rate),
            "ndcg_at_k": MetadataValue.float(ndcg_at_k),
            "best_alpha": MetadataValue.float(best_alpha or 0.0),
            "top_k": MetadataValue.int(RECS_TOP_K),
        }
    )
