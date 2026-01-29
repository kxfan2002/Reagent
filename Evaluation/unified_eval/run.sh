#!/bin/bash
################################################################################
# Unified Evaluation Script
#
# This script runs evaluation on any dataset by simply changing the DATASET variable.
# It automatically configures tools and prompts based on the dataset configuration.
#
# Usage:
#   bash run.sh
#
# Configuration:
#   - Set DATASET to the name of the dataset you want to evaluate
#   - Available datasets: gaia, gaia_text, math500, aime24, aime25, gsm8k,
#     2wiki, hotpotqa, musique, bamboogle, webwalker, hle, xbench
################################################################################

set -e
set -o pipefail

################################################################################
# Environment Configuration
################################################################################

# NCCL configuration for distributed computing
export NCCL_IB_TC=16
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=eth0
export PYTHONDONTWRITEBYTECODE=1
export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_DISABLE=1

################################################################################
# User Configuration Section - MODIFY THESE VALUES
################################################################################

# ============ Dataset Configuration ============
# Change this to evaluate different datasets
# Available: gaia, gaia_text, math500, aime24, aime25, gsm8k, 2wiki, 2wiki200,
#            hotpotqa, hotpotqa1000, hotpotqa_full, musique, musique_full,
#            bamboogle, webwalker, webwalker200, hle, xbench
DATASET="gaia"

# ============ Model Configuration ============
# Path to your model (set via environment variable or modify here)
# export MODEL_PATH="/path/to/your/model"
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

# ============ Output Configuration ============
# Base directory for output results
OUTPUT_PATH="${OUTPUT_PATH:-./output}"

# ============ Evaluation Parameters ============
ROLLOUT_COUNT=3          # Number of rollouts per question (avg@N)
TEMPERATURE=0.6          # Sampling temperature
MAX_WORKERS=16           # Number of concurrent workers (adjust based on resources)
JUDGE_ENGINE="deepseekchat"  # Judge engine: deepseekchat or geminiflash

# ============ Server Configuration ============
# vLLM server ports (modify based on your setup)
PLANNING_PORTS="6001 6002 6003 6004 6005 6006 6007 6008"

# ============ Data Paths ============
# Directory containing evaluation data files
EVAL_DATA_DIR="../eval_data"

# File base path for datasets with file attachments (e.g., GAIA)
FILE_BASE_PATH="${FILE_BASE_PATH:-/path/to/eval_data/gaia}"
export FILE_BASE_PATH

# Image base path for datasets with images
IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-/path/to/eval_data/gaia}"
export IMAGE_BASE_PATH

################################################################################
# API Keys and External Services
################################################################################

## Search API Keys
export SERPER_KEY_ID="${SERPER_KEY_ID:-your_serper_key}"
export SEARCH_API_KEY="${SEARCH_API_KEY:-your_search_key}"
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_key}"

## DeepSeek API configuration (for judge engine)
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-your_deepseek_key}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"

## Gemini API configuration
export GEMINI_API_KEY="${GEMINI_API_KEY:-your_gemini_key}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://api.gemini.com}"

## OpenAI API configuration (for browse tool)
export API_KEY="${API_KEY:-your_openai_key}"
export API_BASE="${API_BASE:-https://api.openai.com/v1}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-gpt-3.5-turbo-0125}"

################################################################################
# Proxy Configuration (Optional)
################################################################################

# Uncomment and configure if you need proxy for external API calls
# PROXY="http://your.proxy.server:port"
# export http_proxy=$PROXY
# export https_proxy=$PROXY
# export HTTP_PROXY=$PROXY
# export HTTPS_PROXY=$PROXY
# export NO_PROXY="localhost,127.0.0.1,::1,.sankuai.com"
# export no_proxy="localhost,127.0.0.1,::1,.sankuai.com"

################################################################################
# Advanced Configuration
################################################################################

# PyTorch configuration
export TORCH_COMPILE_CACHE_DIR="./cache"

# Conda environment (modify based on your setup)
CONDA_ENV="${CONDA_ENV:-vllm}"

################################################################################
# Script Logic - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
################################################################################

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment: ${CONDA_ENV}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
else
    echo "Warning: conda not found, assuming environment is already set up"
fi

# Validate MODEL_PATH
if [ ! -d "$MODEL_PATH" ] && [ "$MODEL_PATH" != "doubao-deepseek-v3.1" ]; then
    echo "Warning: MODEL_PATH does not exist or is not a directory: $MODEL_PATH"
    echo "If this is an API model name, you can ignore this warning."
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p "$OUTPUT_PATH"
mkdir -p ./vllm_logs

# Extract model name for logging
MODEL_BASENAME=$(basename "$MODEL_PATH")

echo ""
echo "========================================="
echo "Unified Evaluation Script"
echo "========================================="
echo "Dataset:          $DATASET"
echo "Model:            $MODEL_PATH"
echo "Output:           $OUTPUT_PATH"
echo "Rollout count:    $ROLLOUT_COUNT"
echo "Max workers:      $MAX_WORKERS"
echo "Temperature:      $TEMPERATURE"
echo "Judge engine:     $JUDGE_ENGINE"
echo "vLLM ports:       $PLANNING_PORTS"
echo "========================================="
echo ""

################################################################################
# Check vLLM Server Status
################################################################################

echo "Checking vLLM server status..."
PORTS_ARRAY=($PLANNING_PORTS)
ALL_SERVERS_READY=true

for port in "${PORTS_ARRAY[@]}"; do
    if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
        echo "✓ Server on port $port is ready"
    else
        echo "✗ Server on port $port is NOT ready"
        ALL_SERVERS_READY=false
    fi
done

if [ "$ALL_SERVERS_READY" = false ]; then
    echo ""
    echo "Warning: Not all vLLM servers are ready!"
    echo "Please start the vLLM servers before running evaluation."
    echo "You can start servers using start_server.sh or manually."
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

################################################################################
# Run Evaluation
################################################################################

echo ""
echo "Starting evaluation..."
echo ""

# Build command with proper argument handling
CMD="python3 -u run.py \
    --dataset \"$DATASET\" \
    --output \"$OUTPUT_PATH\" \
    --max_workers $MAX_WORKERS \
    --model \"$MODEL_PATH\" \
    --temperature $TEMPERATURE \
    --roll_out_count $ROLLOUT_COUNT \
    --eval_data_dir \"$EVAL_DATA_DIR\" \
    --planning_ports $PLANNING_PORTS \
    --auto_judge \
    --judge_engine $JUDGE_ENGINE"

# Add multi-worker support if configured
if [ -n "${WORLD_SIZE}" ] && [ -n "${RANK}" ]; then
    CMD="$CMD --total_splits ${WORLD_SIZE} --worker_split $((${RANK} + 1))"
fi

# Log file
LOG_FILE="./vllm_logs/output_${MODEL_BASENAME}_${DATASET}.log"

echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo ""

# Execute
eval "$CMD" 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "========================================="
echo "Results saved to: $OUTPUT_PATH/${MODEL_BASENAME}_unified/$DATASET/"
echo "Log file: $LOG_FILE"
echo "========================================="

