#!/bin/bash
set -e
mkdir -p jars

curl -fSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    -o jars/hadoop-aws-3.3.4.jar

curl -fSL https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
    -o jars/aws-java-sdk-bundle-1.12.262.jar

curl -fSL https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.8/postgresql-42.7.8.jar \
    -o jars/postgresql-42.7.8.jar

echo "Done."