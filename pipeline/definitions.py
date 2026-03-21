from dagster import Definitions, load_assets_from_modules
 
from pipeline.assets import ingest, bronze, silver, gold, evaluate
from pipeline.resources.spark import SparkSessionResource

all_assets = load_assets_from_modules([ingest, bronze, silver, gold, evaluate])
 
defs = Definitions(
    assets=all_assets,
    resources={
        "spark_resource": SparkSessionResource(
            app_name="FilmRecommendation",
            master_url="spark://spark-master:7077",
        ),
    },
)