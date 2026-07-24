#!/bin/bash
# =============================================================================
# V6-4 Verification Script: Supervisor Auto-Restart of ClawVault
# =============================================================================
#
# This script verifies that when the ClawVault process is killed inside a
# Docker container, Supervisor automatically restarts it within 5 seconds.
#
# Usage:
#   Run inside the Docker container:
#     docker exec <container> /bin/bash /app/scripts/verify_supervisor_restart.sh
#
#   Or during docker run:
#     docker run --rm aegisrouter /bin/bash /app/scripts/verify_supervisor_restart.sh
#
# Prerequisites:
#   - supervisord is running (it's the container's CMD)
#   - ClawVault process is in RUNNING state
#
# Exit codes:
#   0 - ClawVault recovered within 5 seconds (PASS)
#   1 - ClawVault did NOT recover within 5 seconds (FAIL)
#   2 - Prerequisites not met (supervisord not running, etc.)
# =============================================================================

set -euo pipefail

# Configuration
MAX_WAIT_SECONDS=5
POLL_INTERVAL=0.5
PROGRAM_NAME="clawvault"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# ---------------------------------------------------------------------------
# Step 1: Verify supervisord is running
# ---------------------------------------------------------------------------
log_info "Step 1: Checking if supervisord is running..."

if ! command -v supervisorctl &> /dev/null; then
    log_error "supervisorctl not found. Is supervisor installed?"
    exit 2
fi

if ! supervisorctl status &> /dev/null; then
    log_error "supervisord is not running or not reachable."
    exit 2
fi

log_info "supervisord is running."

# ---------------------------------------------------------------------------
# Step 2: Verify ClawVault is currently RUNNING
# ---------------------------------------------------------------------------
log_info "Step 2: Checking if ${PROGRAM_NAME} is RUNNING..."

STATUS=$(supervisorctl status ${PROGRAM_NAME} 2>/dev/null || true)

if echo "$STATUS" | grep -q "RUNNING"; then
    log_info "${PROGRAM_NAME} is RUNNING."
else
    log_warn "${PROGRAM_NAME} is not in RUNNING state: $STATUS"
    log_info "Waiting up to 10s for initial startup..."

    STARTUP_WAIT=0
    while [ $STARTUP_WAIT -lt 10 ]; do
        STATUS=$(supervisorctl status ${PROGRAM_NAME} 2>/dev/null || true)
        if echo "$STATUS" | grep -q "RUNNING"; then
            log_info "${PROGRAM_NAME} reached RUNNING state after ${STARTUP_WAIT}s."
            break
        fi
        sleep 1
        STARTUP_WAIT=$((STARTUP_WAIT + 1))
    done

    if ! echo "$STATUS" | grep -q "RUNNING"; then
        log_error "${PROGRAM_NAME} never reached RUNNING state. Cannot proceed."
        echo "Current status: $STATUS"
        exit 2
    fi
fi

# Record the original PID
ORIGINAL_PID=$(supervisorctl status ${PROGRAM_NAME} | awk '{print $4}' | tr -d ',')
log_info "Original PID: ${ORIGINAL_PID}"

# ---------------------------------------------------------------------------
# Step 3: Kill the ClawVault process
# ---------------------------------------------------------------------------
log_info "Step 3: Killing ${PROGRAM_NAME} process (PID: ${ORIGINAL_PID})..."

# Use kill -9 to simulate an unexpected crash
kill -9 "${ORIGINAL_PID}" 2>/dev/null || true

log_info "Kill signal sent. Starting timer..."

# ---------------------------------------------------------------------------
# Step 4: Measure recovery time
# ---------------------------------------------------------------------------
log_info "Step 4: Waiting for ${PROGRAM_NAME} to restart (max ${MAX_WAIT_SECONDS}s)..."

START_TIME=$(date +%s%N)
ELAPSED=0
RECOVERED=false

while (( $(echo "$ELAPSED < $MAX_WAIT_SECONDS" | bc -l) )); do
    sleep ${POLL_INTERVAL}

    STATUS=$(supervisorctl status ${PROGRAM_NAME} 2>/dev/null || true)

    if echo "$STATUS" | grep -q "RUNNING"; then
        # Verify it's a NEW PID (not the old one somehow still reporting)
        NEW_PID=$(echo "$STATUS" | awk '{print $4}' | tr -d ',')
        if [ "$NEW_PID" != "$ORIGINAL_PID" ]; then
            END_TIME=$(date +%s%N)
            ELAPSED_NS=$((END_TIME - START_TIME))
            ELAPSED_MS=$((ELAPSED_NS / 1000000))
            ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
            RECOVERED=true
            break
        fi
    fi

    CURRENT_TIME=$(date +%s%N)
    ELAPSED_NS=$((CURRENT_TIME - START_TIME))
    ELAPSED=$(echo "scale=2; $ELAPSED_NS / 1000000000" | bc)
done

# ---------------------------------------------------------------------------
# Step 5: Report results
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  V6-4 SUPERVISOR RESTART VERIFICATION"
echo "=============================================="

if [ "$RECOVERED" = true ]; then
    log_info "RESULT: PASS"
    log_info "${PROGRAM_NAME} recovered in ${ELAPSED_S} seconds"
    log_info "Original PID: ${ORIGINAL_PID} → New PID: ${NEW_PID}"
    echo ""
    echo "  ✓ ClawVault auto-restarted within ${MAX_WAIT_SECONDS}s budget"
    echo "  ✓ Recovery time: ${ELAPSED_S}s"
    echo ""
    exit 0
else
    FINAL_STATUS=$(supervisorctl status ${PROGRAM_NAME} 2>/dev/null || echo "UNKNOWN")
    log_error "RESULT: FAIL"
    log_error "${PROGRAM_NAME} did NOT recover within ${MAX_WAIT_SECONDS} seconds"
    log_error "Final status: ${FINAL_STATUS}"
    echo ""
    echo "  ✗ ClawVault failed to auto-restart within ${MAX_WAIT_SECONDS}s"
    echo "  Current status: ${FINAL_STATUS}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check supervisord logs: cat /tmp/supervisord.log"
    echo "  2. Check autorestart setting: grep autorestart /etc/supervisord.conf"
    echo "  3. Check if startretries exhausted: supervisorctl tail ${PROGRAM_NAME}"
    echo ""
    exit 1
fi
