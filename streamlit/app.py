import os
import sys
from typing import Literal
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession, functions as F

from pipeline.resources.spark import SparkSessionResource


BRONZE_MOVIES_PATH = "s3a://bronze/movies/"
BRONZE_RATINGS_PATH = "s3a://bronze/ratings/"
HYBRID_OUT_PATH = "s3a://gold/recommendations/hybrid/"
ALS_MODEL_PATH = "s3a://gold/als_model/"
USER_MEANS_PATH = "s3a://gold/user_means/"

DEFAULT_TOP_K = 10
ALS_EXPANSION_FACTOR = 6
ALS_MIN_CANDIDATES = 100


def _genres_to_text(genres) -> str:
    if genres is None:
        return ""
    if isinstance(genres, float) and pd.isna(genres):
        return ""
    if isinstance(genres, np.ndarray):
        genres = genres.tolist()
    if isinstance(genres, (list, tuple, set)):
        return ", ".join(str(genre) for genre in genres if genre)
    return str(genres)


def _genres_to_list(genres) -> list[str]:
    if genres is None:
        return []
    if isinstance(genres, float) and pd.isna(genres):
        return []
    if isinstance(genres, np.ndarray):
        genres = genres.tolist()
    if isinstance(genres, (list, tuple, set)):
        return [str(genre) for genre in genres if genre]
    return [str(genres)]


@st.cache_data(ttl=300, show_spinner=False)
def get_movies_catalog_pd() -> pd.DataFrame:
    spark = get_spark()
    movies_df = spark.read.parquet(BRONZE_MOVIES_PATH).select(
        F.col("id").alias("movieId"),
        "title",
        "genre_list",
        "release_date",
        "runtime",
        "overview",
    )
    movies_pd = movies_df.toPandas()
    movies_pd["genre_text"] = movies_pd["genre_list"].apply(_genres_to_text)
    return movies_pd


@st.cache_resource(show_spinner=False)
def get_spark() -> SparkSession:
    resource = SparkSessionResource(
        app_name="FilmRecommendationStreamlit",
        master_url=os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077"),
    )
    return resource.get_session()


@st.cache_resource(show_spinner=False)
def get_als_model() -> ALSModel:
    return ALSModel.load(ALS_MODEL_PATH)


@st.cache_data(ttl=300, show_spinner=False)
def get_movies_pd() -> pd.DataFrame:
    return get_movies_catalog_pd().drop(columns=["genre_list"])


@st.cache_data(ttl=300, show_spinner=False)
def get_user_catalog() -> dict[str, list[int]]:
    spark = get_spark()
    ratings_users = [
        row["userId"]
        for row in spark.read.parquet(BRONZE_RATINGS_PATH)
        .select("userId")
        .distinct()
        .orderBy("userId")
        .toLocalIterator()
    ]

    try:
        hybrid_users = [
            row["userId"]
            for row in spark.read.parquet(HYBRID_OUT_PATH)
            .select("userId")
            .distinct()
            .orderBy("userId")
            .toLocalIterator()
        ]
    except Exception:
        hybrid_users = []
    return {
        "all_users": ratings_users,
        "hybrid_users": hybrid_users,
    }


def _format_results(results_pd: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    if results_pd.empty:
        return results_pd

    columns = [
        "movieId",
        "title",
        "release_date",
        "runtime",
        "genre_text",
        "overview",
    ]
    if score_col:
        columns.insert(2, score_col)
    formatted = results_pd[columns].copy()
    if score_col:
        formatted[score_col] = formatted[score_col].astype(float).round(4)
    formatted["runtime"] = formatted["runtime"].fillna(0).astype(float).round(0)
    formatted["overview"] = formatted["overview"].fillna("").str.slice(0, 240)
    return formatted


def load_hybrid_recommendations(user_id: int, top_k: int) -> pd.DataFrame:
    spark = get_spark()
    hybrid_df = spark.read.parquet(HYBRID_OUT_PATH).select(
        "userId",
        "movieId",
        "hybrid_score",
        "collab_score",
        "content_score",
    )
    movies_df = spark.read.parquet(BRONZE_MOVIES_PATH).select(
        F.col("id").alias("movieId"),
        "title",
        "genre_list",
        "release_date",
        "runtime",
        "overview",
    )

    results_df = (
        hybrid_df.filter(F.col("userId") == user_id)
        .join(movies_df, "movieId", "left")
        .orderBy(
            F.desc("hybrid_score"),
            F.desc("collab_score"),
            F.desc("content_score"),
            F.asc("movieId"),
        )
        .limit(top_k)
    )

    results_pd = results_df.toPandas()
    if results_pd.empty:
        return results_pd

    results_pd["genre_text"] = results_pd["genre_list"].apply(_genres_to_text)
    return _format_results(results_pd.drop(columns=["genre_list"]), None)


def load_als_recommendations(user_id: int, top_k: int) -> pd.DataFrame:
    spark = get_spark()
    model = get_als_model()

    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    user_means_df = spark.read.parquet(USER_MEANS_PATH)
    seen_df = (
        spark.read.parquet(BRONZE_RATINGS_PATH)
        .filter(F.col("userId") == user_id)
        .select("userId", "movieId")
        .distinct()
    )
    movies_df = spark.read.parquet(BRONZE_MOVIES_PATH).select(
        F.col("id").alias("movieId"),
        "title",
        "genre_list",
        "release_date",
        "runtime",
        "overview",
    )

    candidate_limit = max(top_k * ALS_EXPANSION_FACTOR, ALS_MIN_CANDIDATES)
    results_df = (
        model.recommendForUserSubset(user_df, candidate_limit)
        .select("userId", F.explode("recommendations").alias("rec"))
        .select(
            "userId",
            F.col("rec.movieId").alias("movieId"),
            F.col("rec.rating").alias("prediction_norm"),
        )
        .join(user_means_df, "userId", "left")
        .join(seen_df, ["userId", "movieId"], "left_anti")
        .withColumn("predicted_rating", F.col("prediction_norm") + F.col("user_mean"))
        .join(movies_df, "movieId", "left")
        .orderBy(F.desc("predicted_rating"), F.asc("movieId"))
        .limit(top_k)
    )

    results_pd = results_df.toPandas()
    if results_pd.empty:
        return results_pd

    results_pd["genre_text"] = results_pd["genre_list"].apply(_genres_to_text)
    return _format_results(results_pd.drop(columns=["genre_list"]), None)


def load_user_history(user_id: int, limit: int = 10) -> pd.DataFrame:
    spark = get_spark()
    ratings_df = spark.read.parquet(BRONZE_RATINGS_PATH).select(
        "userId", "movieId", "rating", "timestamp"
    )
    movies_df = spark.read.parquet(BRONZE_MOVIES_PATH).select(
        F.col("id").alias("movieId"),
        "title",
        "release_date",
        "genre_list",
    )
    history_df = (
        ratings_df.filter(F.col("userId") == user_id)
        .join(movies_df, "movieId", "left")
        .orderBy(F.desc("timestamp"))
        .limit(limit)
    )
    history_pd = history_df.toPandas()
    if history_pd.empty:
        return history_pd

    history_pd["genre_text"] = history_pd["genre_list"].apply(_genres_to_text)
    return history_pd[["movieId", "title", "rating", "release_date", "genre_text"]]


@st.cache_data(ttl=300, show_spinner=False)
def get_popular_movies_pd() -> pd.DataFrame:
    spark = get_spark()
    ratings_df = spark.read.parquet(BRONZE_RATINGS_PATH).select("movieId", "rating")
    movies_df = spark.read.parquet(BRONZE_MOVIES_PATH).select(
        F.col("id").alias("movieId"),
        "title",
        "genre_list",
        "release_date",
        "runtime",
        "overview",
    )
    popular_df = (
        ratings_df.groupBy("movieId")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("avg_rating"),
        )
        .join(movies_df, "movieId", "inner")
    )

    popular_pd = popular_df.toPandas()
    popular_pd["genre_text"] = popular_pd["genre_list"].apply(_genres_to_text)
    popular_pd["popularity_score"] = (
        popular_pd["avg_rating"].fillna(0.0) * np.log1p(popular_pd["rating_count"].fillna(0))
    )
    return popular_pd


def load_cold_start_recommendations(selected_genres: list[str], seed_movie_ids: list[int], top_k: int) -> pd.DataFrame:
    popular_pd = get_popular_movies_pd().copy()
    preferred_genres = set(selected_genres)

    if seed_movie_ids:
        seed_rows = popular_pd[popular_pd["movieId"].isin(seed_movie_ids)]
        for genres in seed_rows["genre_list"].tolist():
            preferred_genres.update(_genres_to_list(genres))

    popular_pd["genre_overlap"] = popular_pd["genre_list"].apply(
        lambda genres: len(preferred_genres.intersection(_genres_to_list(genres)))
    )
    popular_pd["seed_penalty"] = popular_pd["movieId"].isin(seed_movie_ids).astype(int)
    ranked_pd = popular_pd.sort_values(
        by=["genre_overlap", "popularity_score", "avg_rating", "rating_count", "title"],
        ascending=[False, False, False, False, True],
    )
    ranked_pd = ranked_pd[ranked_pd["seed_penalty"] == 0]

    if preferred_genres:
        ranked_pd = ranked_pd[
            (ranked_pd["genre_overlap"] > 0)
            | (ranked_pd["rating_count"] >= ranked_pd["rating_count"].quantile(0.8))
        ]

    results_pd = ranked_pd.head(top_k).copy()
    if results_pd.empty:
        return results_pd

    return _format_results(results_pd.drop(columns=["genre_list"]), None)


def render_metric_cards() -> None:
    users = get_user_catalog()
    movies_pd = get_movies_pd()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tong user", len(users["all_users"]))
    col2.metric("User co hybrid", len(users["hybrid_users"]))
    col3.metric("Tong phim", len(movies_pd))


def render_serving_guidance(selected_mode: Literal["hybrid", "als_model", "cold_start"]) -> None:
    if selected_mode == "hybrid":
        st.info(
            "Hybrid phu hop khi da materialize `gold_hybrid_recommendations` "
            "va user co du lich su tuong tac de ket hop collaborative + content."
        )
    elif selected_mode == "cold_start":
        st.info(
            "Cold-start phu hop cho user moi chua co `userId` hoac chua co lich su. "
            "App se ket hop phim pho bien voi uu tien theo the loai va seed movie ban chon."
        )
    else:
        st.info(
            "ALSModel phu hop cho phuc vu online nhanh hon, co the suy luan "
            "truc tiep cho user da duoc train ma khong can cho output hybrid co san."
        )

    st.caption(
        "Cold-start user se khong co trong danh sach `userId` hien tai. Huong xu ly "
        "hop ly la hoi nhanh so thich ban dau nhu the loai, 3-5 phim yeu thich, "
        "hoac ngon ngu, sau do goi y bang content-based/popular; khi user da co "
        "du tuong tac thi moi chuyen sang `hybrid` hoac `als_model`."
    )


def render_results_table(results_pd: pd.DataFrame, score_label: str | None = None) -> None:
    if results_pd.empty:
        st.warning("Khong tim thay recommendation cho user nay voi che do da chon.")
        return

    if score_label:
        results_pd = results_pd.rename(columns={score_label: "score"})

    st.dataframe(results_pd, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Film Recommendation Demo", layout="wide")

    st.title("Film Recommendation Demo")

    render_metric_cards()

    user_catalog = get_user_catalog()
    hybrid_users = user_catalog["hybrid_users"]
    all_users = user_catalog["all_users"]

    st.sidebar.header("Cau hinh")
    serving_mode = st.sidebar.radio(
        "Che do du doan",
        options=["hybrid", "als_model", "cold_start"],
        format_func=lambda value: {
            "hybrid": "Hybrid recommendation",
            "als_model": "ALSModel realtime",
            "cold_start": "Cold start",
        }[value],
    )
    top_k = st.sidebar.slider("So goi y", min_value=5, max_value=20, value=DEFAULT_TOP_K, step=1)

    render_serving_guidance(serving_mode)
    if serving_mode == "cold_start":
        movies_catalog = get_movies_catalog_pd()
        all_genres = sorted(
            {
                genre
                for genres in movies_catalog["genre_list"].tolist()
                for genre in (
                    genres.tolist() if isinstance(genres, np.ndarray) else (genres or [])
                )
                if genre
            }
        )
        movie_options = (
            movies_catalog.sort_values("title")[["movieId", "title", "release_date"]]
            .assign(
                movie_label=lambda df: df.apply(
                    lambda row: f"{row['title']} ({row['release_date'] if pd.notna(row['release_date']) else 'unknown'}) - {row['movieId']}",
                    axis=1,
                )
            )
        )
        movie_label_to_id = dict(zip(movie_options["movie_label"], movie_options["movieId"]))

        selected_genres = st.sidebar.multiselect(
            "The loai yeu thich",
            options=all_genres,
            placeholder="Chon 1 hoac nhieu the loai",
        )
        selected_seed_labels = st.sidebar.multiselect(
            "Phim ban da thich",
            options=movie_options["movie_label"].tolist(),
            placeholder="Co the bo qua neu chua biet",
        )
        selected_seed_ids = [movie_label_to_id[label] for label in selected_seed_labels]

        st.subheader("Goi y cho user moi")
        st.caption(
            "Neu chua co `userId`, hay chon vai the loai hoac phim mau. "
            "He thong se uu tien phim pho bien nhung gan voi so thich ban dau."
        )
        with st.spinner("Dang tao goi y cold-start..."):
            results_pd = load_cold_start_recommendations(selected_genres, selected_seed_ids, top_k)
        render_results_table(results_pd, None)
        return

    default_users = hybrid_users if serving_mode == "hybrid" else all_users
    default_index = 0 if default_users else None
    if not default_users:
        st.error("Chua co user kha dung trong data lake.")
        return

    if serving_mode == "hybrid" and not hybrid_users:
        st.warning(
            "Chua tim thay output `s3a://gold/recommendations/hybrid/`. "
            "Hay materialize asset `gold_hybrid_recommendations` hoac chuyen sang `als_model`."
        )
        return

    selected_user = st.sidebar.selectbox("Chon userId", options=default_users, index=default_index)

    history_col, result_col = st.columns([1, 2])

    with history_col:
        st.subheader(f"Lich su gan day cua user {selected_user}")
        history_pd = load_user_history(int(selected_user))
        if history_pd.empty:
            st.caption("Khong co lich su rating.")
        else:
            st.dataframe(history_pd, use_container_width=True, hide_index=True)

    with result_col:
        st.subheader(f"Top {top_k} du doan cho user {selected_user}")
        with st.spinner("Dang tai recommendation..."):
            if serving_mode == "hybrid":
                results_pd = load_hybrid_recommendations(int(selected_user), top_k)
                score_col = None
            else:
                results_pd = load_als_recommendations(int(selected_user), top_k)
                score_col = None
        render_results_table(results_pd, score_col)


if __name__ == "__main__":
    main()
