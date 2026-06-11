#!/bin/bash
# run.sh -- System info collection script for Linux/Mac.

FILTER="${1:-all}"

case "$FILTER" in
  cpu)
    echo "CPU: $(nproc) cores"
    ;;
  memory)
    if command -v free >/dev/null 2>&1; then
      free -h
    else
      echo "Memory info unavailable (free not found)"
    fi
    ;;
  disk)
    if command -v df >/dev/null 2>&1; then
      df -h
    else
      echo "Disk info unavailable (df not found)"
    fi
    ;;
  all)
    echo "=== CPU ==="
    echo "CPU: $(nproc) cores"
    echo ""
    echo "=== Memory ==="
    if command -v free >/dev/null 2>&1; then
      free -h
    else
      echo "Memory info unavailable"
    fi
    echo ""
    echo "=== Disk ==="
    if command -v df >/dev/null 2>&1; then
      df -h
    else
      echo "Disk info unavailable"
    fi
    ;;
  *)
    echo "Unknown filter: $FILTER. Use cpu, memory, disk, or all."
    ;;
esac
