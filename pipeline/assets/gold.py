from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, row_number, percent_rank
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pipeline.assets.bronze import bronze_ratings
from pipeline.assets.silver import silver_genres_tfidf, silver_synopsis_embeddings
from pipeline.resources.spark import SparkSessionResource
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import numpy as np

@asset(deps=[bronze_ratings])
def gold_als_model(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    # Đọc bronze ratings
    ratings_df = spark.read.parquet("s3a://bronze/ratings/")

    # Train/test split theo timestamp (không random để tránh data leakage)
    # Sort by timestamp, dùng percent_rank để split 80/20
    window = Window.orderBy("timestamp")
    split_df = ratings_df.withColumn("rank", percent_rank().over(window))

    train_df = split_df.filter(col("rank") < 0.8).drop("rank")
    test_df = split_df.filter(col("rank") >= 0.8).drop("rank")

    # ALS training
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)

    # Evaluate trên test
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    context.log.info(f"ALS RMSE: {rmse}")

    # Lưu model
    model.write().overwrite().save("s3a://gold/als_model/")

    return MaterializeResult(
        metadata={
            "rmse": MetadataValue.float(rmse),
            "model_path": MetadataValue.text("s3a://gold/als_model/"),
        }
    )

@asset(deps=[silver_genres_tfidf, silver_synopsis_embeddings])
def gold_qdrant_upsert(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    # Đọc movies bronze
    movies_df = spark.read.parquet("s3a://bronze/movies/")
    # Đọc silver features
    genres_df = spark.read.parquet("s3a://silver/genres_tfidf/")
    synopsis_df = spark.read.parquet("s3a://silver/synopsis_embeddings/")

    # Join để có full data
    full_df = movies_df.join(genres_df, "id").join(synopsis_df, "id")

    # Qdrant client
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

    # Tạo collection nếu chưa có
    client.recreate_collection(
        collection_name="movies",
        vectors_config={
            "genre_tfidf": VectorParams(size=1000, distance=Distance.COSINE),
            "synopsis_embedding": VectorParams(size=384, distance=Distance.COSINE),  # all-MiniLM-L6-v2 is 384 dim
        }
    )

    # Upsert theo batch 500
    points = []
    batch_size = 500
    count = 0

    for row in full_df.collect():
        payload = {
            "id": row["id"],
            "title": row["title"],
            "genre_list": row["genre_list"],
            "release_date": row["release_date"],
            "runtime": row["runtime"],
            "overview": row["overview"],
        }
        vectors = {
            "genre_tfidf": row["genre_tfidf"].toArray().tolist(),  # SparseVector to list
            "synopsis_embedding": row["synopsis_embedding"],
        }
        points.append(PointStruct(id=row["id"], payload=payload, vector=vectors))

        if len(points) >= batch_size:
            client.upsert(collection_name="movies", points=points)
            count += len(points)
            points = []
            context.log.info(f"Upserted {count} points")

    # Upsert remaining
    if points:
        client.upsert(collection_name="movies", points=points)
        count += len(points)

    context.log.info(f"Total upserted: {count}")

    return MaterializeResult(
        metadata={
            "collection": MetadataValue.text("movies"),
            "points_upserted": MetadataValue.int(count),
        }
    )

@asset(deps=[gold_als_model, gold_qdrant_upsert])
def gold_hybrid_recommendations(context: AssetExecutionContext, spark_resource: SparkSessionResource) -> MaterializeResult:
    spark = spark_resource.get_session()

    # Load ALS model
    from pyspark.ml.recommendation import ALSModel
    model = ALSModel.load("s3a://gold/als_model/")

    # Đọc ratings
    ratings_df = spark.read.parquet("s3a://bronze/ratings/")

    # Get collaborative recommendations (ALS predictions)
    collab_recs = model.recommendForAllUsers(10)  # Top 10 per user

    # For simplicity, compute for a sample user, e.g., userId=1
    # In real, loop over users or something
    user_id = 1
    user_collab = collab_recs.filter(col("userId") == user_id).select("recommendations").collect()[0]["recommendations"]

    # Content-based: query Qdrant for similar movies based on user's rated movies
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

    # Get user's rated movies
    user_ratings = ratings_df.filter(col("userId") == user_id).select("movieId", "rating").collect()
    high_rated = [r["movieId"] for r in user_ratings if r["rating"] >= 4.0]

    if high_rated:
        # Use synopsis embedding of first high-rated movie for search
        query_vector = client.retrieve(collection_name="movies", ids=[high_rated[0]])[0].vector["synopsis_embedding"]
        content_results = client.search(
            collection_name="movies",
            query_vector=query_vector,
            vector_name="synopsis_embedding",
            limit=10
        )
        content_movie_ids = [r.id for r in content_results]
    else:
        content_movie_ids = []

    # Hybrid scoring: α=0.6 for collaborative, 0.4 for content
    alpha = 0.6
    hybrid_scores = {}

    # Collaborative scores
    for rec in user_collab:
        hybrid_scores[rec["movieId"]] = {"collab_score": rec["rating"], "content_score": 0.0}

    # Content scores (simplified, assume score from search)
    for i, mid in enumerate(content_movie_ids):
        score = 1.0 / (i + 1)  # Rank-based score
        if mid in hybrid_scores:
            hybrid_scores[mid]["content_score"] = score
        else:
            hybrid_scores[mid] = {"collab_score": 0.0, "content_score": score}

    # Combine
    for mid in hybrid_scores:
        collab = hybrid_scores[mid]["collab_score"]
        content = hybrid_scores[mid]["content_score"]
        hybrid_scores[mid]["hybrid_score"] = alpha * collab + (1 - alpha) * content

    # Sort and save top 10
    top_recs = sorted(hybrid_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:10]

    # Save to file or something
    recs_df = spark.createDataFrame([
        (user_id, mid, scores["collab_score"], scores["content_score"], scores["hybrid_score"])
        for mid, scores in top_recs
    ], ["userId", "movieId", "collab_score", "content_score", "hybrid_score"])

    recs_df.write.mode("overwrite").parquet(f"s3a://gold/recommendations/user_{user_id}/")

    context.log.info(f"Hybrid recommendations for user {user_id} saved.")

    return MaterializeResult(
        metadata={
            "user_id": MetadataValue.int(user_id),
            "output_path": MetadataValue.text(f"s3a://gold/recommendations/user_{user_id}/"),
        }
    )