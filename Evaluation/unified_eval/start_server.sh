#!/bin/bash
################################################################################
# vLLM Server Startup Script
#
# This script starts multiple vLLM server instances for model serving.
# Each server runs on a dedicated GPU.
#
# Usage:
#   bash start_server.sh
################################################################################

set -e

################################################################################
# Configuration
################################################################################

# Model path
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

# Server configuration
SERVER_HOST="0.0.0.0"
BACKEND_PORTS=(6001 6002 6003 6004 6005 6006 6007 6008)
GPU_DEVICES=(0 1 2 3 4 5 6 7)

# Log directory
LOG_DIR="./vllm_logs"

# Startup wait time
STARTUP_WAIT_TIME=60  # seconds

# Health check timeout
HEALTH_CHECK_TIMEOUT=600  # seconds

################################################################################
# Validate Configuration
################################################################################

if [ ! -d "$MODEL_PATH" ] && [ "$MODEL_PATH" = "/path/to/your/model" ]; then
    echo "Error: Please set MODEL_PATH to your model directory"
    echo "You can either:"
    echo "  1. Edit this script and set MODEL_PATH variable"
    echo "  2. Set environment variable: export MODEL_PATH=/path/to/model"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_DIR}"

################################################################################
# Start vLLM Servers
################################################################################

echo "========================================="
echo "Starting vLLM Servers"
echo "========================================="
echo "Model path: ${MODEL_PATH}"
echo "Number of servers: ${#BACKEND_PORTS[@]}"
echo "Log directory: ${LOG_DIR}"
echo ""

# Start each server
for i in "${!BACKEND_PORTS[@]}"; do
    port=${BACKEND_PORTS[$i]}
    gpu=${GPU_DEVICES[$i]}
    log_file="${LOG_DIR}/server_${port}.log"
    
    echo "Starting server ${i}: GPU=${gpu}, Port=${port}"
    
    CUDA_VISIBLE_DEVICES=${gpu} vllm serve "${MODEL_PATH}" \
        --host ${SERVER_HOST} \
        --port ${port} \
        --disable-log-requests \
        > "${log_file}" 2>&1 &
    
    SERVER_PID=$!
    echo "  - Server PID: ${SERVER_PID}"
    echo "  - Log file: ${log_file}"
done

echo ""
echo "All servers launched. Waiting ${STARTUP_WAIT_TIME} seconds for initialization..."
sleep ${STARTUP_WAIT_TIME}

################################################################################
# Health Check
################################################################################

echo ""
echo "Checking server health..."
echo ""

start_time=$(date +%s)
all_healthy=false

while true; do
    all_ready=true
    
    for port in "${BACKEND_PORTS[@]}"; do
        if curl -s -f http://localhost:${port}/v1/models > /dev/null 2>&1; then
            echo "✓ Server on port ${port} is ready"
        else
            echo "✗ Server on port ${port} is not ready yet"
            all_ready=false
        fi
    done
    
    if [ "$all_ready" = true ]; then
        all_healthy=true
        break
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $HEALTH_CHECK_TIMEOUT ]; then
        echo ""
        echo "Error: Health check timeout after ${HEALTH_CHECK_TIMEOUT} seconds"
        break
    fi
    
    echo ""
    echo "Waiting for servers to be ready... (${elapsed}s elapsed)"
    sleep 10
done

################################################################################
# Summary
################################################################################

echo ""
echo "========================================="
if [ "$all_healthy" = true ]; then
    echo "All servers are ready!"
else
    echo "Warning: Not all servers are ready"
fi
echo "========================================="
echo ""
echo "Server endpoints:"
for port in "${BACKEND_PORTS[@]}"; do
    echo "  - http://localhost:${port}/v1"
done
echo ""
echo "To check status:"
echo "  curl http://localhost:6001/v1/models"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/server_*.log"
echo ""
echo "To stop all servers:"
echo "  pkill -f 'vllm serve'"
echo ""
echo "========================================="

