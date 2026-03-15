#!/bin/bash
set -e
dagster-webserver -h 0.0.0.0 -p 3000 &
WEBSERVER_PID=$!
dagster-daemon run &
DAEMON_PID=$!

#Ensure both webserver and daemon run correctly
wait -n $WEBSERVER_PID $DAEMON_PID
exit $?