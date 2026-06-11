#!/usr/bin/env bash
# stop.sh -- Stop the 00_smart_app server.
# This script lives in bin/; the app base is the parent directory.

set -e

PROCESS_NAME="00_smart_app"

find_smart_app_pids() {
    ps -eo pid=,comm= | grep -E "[[:space:]]${PROCESS_NAME}$" | awk '{print $1}' || true
}

PIDS="$(find_smart_app_pids)"
if [[ -z "$PIDS" ]]; then
    echo "00_smart_app is not running"
    exit 0
fi

echo "Stopping 00_smart_app (PID(s): ${PIDS//$'\n'/ })..."
kill $PIDS

# Wait up to 5 seconds for graceful shutdown
for i in {1..10}; do
    PIDS="$(find_smart_app_pids)"
    if [[ -z "$PIDS" ]]; then
        break
    fi
    sleep 0.5
done

PIDS="$(find_smart_app_pids)"
if [[ -n "$PIDS" ]]; then
    echo "Force killing..."
    kill -9 $PIDS
fi

echo "Stopped."
