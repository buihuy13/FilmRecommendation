#!/bin/bash
# File: init-db/01_create_databases.sh
# PostgreSQL sẽ tự chạy file này khi container khởi động lần đầu
 
set -e
 
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE $METABASE_DB;
    CREATE DATABASE $HIVE_DB;
 
    GRANT ALL PRIVILEGES ON DATABASE $METABASE_DB TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE $HIVE_DB TO $POSTGRES_USER;
EOSQL