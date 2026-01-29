#!/bin/bash
set -ex

################################################################################
# Configuration Section - Modify these variables for your setup
################################################################################

# Project paths
PROJECT_ROOT="/path/to/your/reagent"  # TODO: Set this to your reagent project root directory
MODEL_PATH="/path/to/your/model"      # TODO: Set this to your base model path (e.g., Qwen2.5-7B)
DATA_DIR="${PROJECT_ROOT}/data"       # Base directory for data files
LOG_DIR="${PROJECT_ROOT}/vllm_logs"   # Directory for training logs

# Experiment configuration
EXPERIMENT_NAME="all-tool-adaptive-reward-score-0.3-critique"  # Name of this training experiment

# Training hyperparameters
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=30
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=11000
LEARNING_RATE=5e-6
REWARD_MODEL_LAMBDA=0.3
CRITIQUE_LAMBDA=0.3

################################################################################
# Environment Setup
################################################################################

# VLLM configuration
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Activate conda environment
conda activate reagent

# Optional proxy configuration (uncomment if needed)
# PROXY="http://127.0.0.1:3128" 
# export http_proxy=$PROXY
# export https_proxy=$PROXY
# export HTTP_PROXY=$PROXY
# export HTTPS_PROXY=$PROXY
# export NO_PROXY="localhost,127.0.0.1,::1"
# export no_proxy="localhost,127.0.0.1,::1"

################################################################################
# API Keys Configuration
################################################################################

# SwanLab for experiment tracking (https://swanlab.cn/)
export SWANLAB_API_KEY=""  # TODO: Set your SwanLab API key for logging

# Search & Browse Tools
export SEARCH_API_KEY=""   # TODO: Set your search API key (e.g., Serper, SerpAPI)
export JINA_API_KEY=""     # TODO: Set your Jina Reader API key for web content extraction

# DeepSeek API configuration (for LLM-based tools)
export DEEPSEEK_API_KEY=""      # TODO: Set your DeepSeek API key
export DEEPSEEK_API_BASE=""     # TODO: Set DeepSeek API base URL (e.g., https://api.deepseek.com)

# Gemini API configuration (alternative LLM provider)
export GEMINI_API_KEY=""        # TODO: Set your Gemini API key
export GEMINI_API_BASE=""       # TODO: Set Gemini API base URL

################################################################################
# Tool Configuration - Image Processing
################################################################################

export IMAGE2TEXT_API_URL=""                  # TODO: Set image-to-text API endpoint (e.g., GPT-4V API)
export IMAGE2TEXT_MODEL="gpt-4.1"            # Model name for image-to-text conversion
export IMAGE_BASE_PATH="${DATA_DIR}/images"  # Base path for image files

################################################################################
# Tool Configuration - Audio Processing
################################################################################

export WHISPER_API="http://127.0.0.1:8000/v1/audio/transcriptions"  # Whisper API endpoint
export WHISPER_MODEL="whisper"                                       # Whisper model name
export MAX_DURATION=30                                               # Maximum audio duration in seconds
export AUDIO_BASE_PATH="${DATA_DIR}/audio"                          # Base path for audio files
export AUDIO_TEMP_DIR="${DATA_DIR}/audio_temp"                      # Temporary directory for audio processing
export AUDIO_TIMEOUT=60                                             # Audio API timeout in seconds
export AUDIO_MAX_RETRY=3                                            # Maximum retries for audio API calls

################################################################################
# Tool Configuration - File Reader
################################################################################

export FILE_BASE_PATH="${DATA_DIR}"     # Base path for file reading tool
export FILEREADER_MAX_CHARS=2000        # Maximum characters to read from a file

################################################################################
# Reward Model Configuration
################################################################################

export REWARD_MODEL_URL="http://127.0.0.1:8001"  # Reward model API endpoint
export ENABLE_REWARD_MODEL=true                   # Enable/disable reward model
export REWARD_MODEL_NAME="reward_model"           # Reward model name
export REWARD_MODEL_LAMBDA=${REWARD_MODEL_LAMBDA} # Weight for reward model score

################################################################################
# Critique Configuration
################################################################################

export CRITIQUE_LAMBDA=${CRITIQUE_LAMBDA}  # Weight for critique-based rewards

# Optional: Use local SwanLab mode (uncomment if needed)
# export SWANLAB_MODE=local

################################################################################
# Training Script Execution
################################################################################

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Change to project root directory
cd "${PROJECT_ROOT}"

# Run training with Hydra configuration
python3 -m examples.reagent.train \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.34 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=13 \
    rllm.stepwise_advantage.enable=False \
    rllm.reward_model.enable=${ENABLE_REWARD_MODEL} \
    rllm.reward_model.url=${REWARD_MODEL_URL} \
    rllm.reward_model.lambda=${REWARD_MODEL_LAMBDA} \
    rllm.rejection_sample.enable=True \
    rllm.two_stage_sampling.enable=True \
    rllm.two_stage_sampling.use_critique=True \
    rllm.critique.enable=True \
    rllm.critique.lambda=${CRITIQUE_LAMBDA} \
    rllm.critique.max_concurrency=32 \
    trainer.total_epochs=3 \
    2>&1 | tee "${LOG_DIR}/output_${EXPERIMENT_NAME}.log"

echo "Training completed! Logs saved to: ${LOG_DIR}/output_${EXPERIMENT_NAME}.log"