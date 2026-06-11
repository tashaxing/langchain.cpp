#!/usr/bin/env bash
# start.sh -- Start the 00_smart_app server.
# This script lives in bin/; the app base is the parent directory.

set -e

PROCESS_NAME="00_smart_app"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="${SCRIPT_DIR}/${PROCESS_NAME}"
CONFIG="${APP_DIR}/config/app_config.xml"
LOGDIR="${APP_DIR}/log"

# Allow override via environment
if [[ -n "$SMART_APP_DIR" ]]; then
    APP_DIR="$SMART_APP_DIR"
    BINARY="${APP_DIR}/bin/${PROCESS_NAME}"
    CONFIG="${APP_DIR}/config/app_config.xml"
    LOGDIR="${APP_DIR}/log"
fi

find_smart_app_pids() {
    ps -eo pid=,comm= | grep -E "[[:space:]]${PROCESS_NAME}$" | awk '{print $1}' || true
}

if [[ ! -f "$BINARY" ]]; then
    echo "Error: binary not found: $BINARY"
    echo "Please build and install the project first."
    exit 1
fi

# Ensure log directory exists
mkdir -p "$LOGDIR"

PIDS="$(find_smart_app_pids)"
if [[ -n "$PIDS" ]]; then
    echo "00_smart_app is already running (PID(s): ${PIDS//$'\n'/ })"
    exit 0
fi

echo "Starting 00_smart_app..."
nohup "$BINARY" --config "$CONFIG" > "${APP_DIR}/smart_app.out" 2>&1 &

sleep 1
PIDS="$(find_smart_app_pids)"
if [[ -z "$PIDS" ]]; then
    echo "Error: 00_smart_app exited immediately."
    exit 1
fi

echo "Started (PID(s): ${PIDS//$'\n'/ })"
echo "Config: $CONFIG"
echo "Logs:   ${LOGDIR}/"
