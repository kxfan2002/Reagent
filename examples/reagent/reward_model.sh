#!/bin/bash
################################################################################
# Reward Model Server Deployment Script
#
# This script starts multiple VLLM server instances for reward model inference
# and a load balancer to distribute requests among them.
#
# Usage:
#   bash reward_model.sh
################################################################################

set -e
set -o pipefail

################################################################################
# Configuration Section
################################################################################

# Model configuration
MODEL_PATH="/path/to/reward_model"  # TODO: Set path to your reward model

# Server configuration
BACKEND_PORTS=(6001 6002 6003 6004)  # Backend server ports
LOAD_BALANCER_PORT=8001              # Load balancer port
SERVER_HOST="127.0.0.1"              # Server host address

# GPU configuration (one GPU per server instance)
GPU_DEVICES=(0 1 2 3)  # GPU device IDs for each server

# Timing configuration
STARTUP_WAIT_TIME=30  # Seconds to wait for servers to start

# Directory configuration
LOG_DIR="./vllm_logs"  # Directory for log files
LOAD_BALANCER_SCRIPT="load_balance.py"  # Load balancer script name

# Conda environment
CONDA_ENV="vllm"  # Conda environment name

################################################################################
# Environment Setup
################################################################################

# NCCL configuration for distributed computing
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_NET=Socket

# OpenMP configuration
export OMP_NUM_THREADS=8

# Other environment variables
export DECORD_EOF_RETRY_MAX=2048001

# Optional: PyTorch CUDA memory configuration
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"

# Activate conda environment
conda activate ${CONDA_ENV}

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

################################################################################
# Start VLLM Backend Servers
################################################################################

echo "Starting VLLM reward model servers..."
echo "Model path: ${MODEL_PATH}"
echo "Number of servers: ${#BACKEND_PORTS[@]}"
echo ""

# Start each VLLM server instance with dedicated GPU
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
done

echo ""
echo "Waiting ${STARTUP_WAIT_TIME} seconds for servers to start..."
sleep ${STARTUP_WAIT_TIME}

################################################################################
# Health Check
################################################################################

echo ""
echo "Checking VLLM server status..."
all_healthy=true

for port in "${BACKEND_PORTS[@]}"; do
    if nc -z ${SERVER_HOST} ${port} 2>/dev/null; then
        echo "✓ Server on port ${port} is running"
    else
        echo "✗ Server on port ${port} may not be ready yet"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo ""
    echo "Warning: Some servers may not be fully started. Check logs if issues persist."
fi

################################################################################
# Start Load Balancer
################################################################################

echo ""
echo "Starting load balancer on port ${LOAD_BALANCER_PORT}..."
python "${LOAD_BALANCER_SCRIPT}" > "${LOG_DIR}/load_balancer.log" 2>&1 &
LB_PID=$!

echo ""
echo "========================================="
echo "All services started!"
echo "========================================="
echo "Load balancer: http://localhost:${LOAD_BALANCER_PORT}"
echo "Backend servers: ${BACKEND_PORTS[*]}"
echo "Load balancer PID: ${LB_PID}"
echo ""
echo "Health check: curl http://localhost:${LOAD_BALANCER_PORT}/health"
echo "View stats:   curl http://localhost:${LOAD_BALANCER_PORT}/stats"
echo ""
echo "Logs:"
echo "  - Load balancer: tail -f ${LOG_DIR}/load_balancer.log"
echo "  - VLLM servers:  tail -f ${LOG_DIR}/server_*.log"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait 
