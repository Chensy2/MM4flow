#!/bin/bash
#SBATCH --job-name=gpr-cls-mm4flow
#SBATCH --time=120:00:00
#SBATCH --partition=big
#SBATCH --gres=shard:A800:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --output=./MM4flow/finetune_log/ITC_Net_A/D_global_pseudo_residual_cls_ps_byte_mm.out
#SBATCH --error=./MM4flow/finetune_log/ITC_Net_A/D_global_pseudo_residual_cls_ps_byte_mm.err

set -euo pipefail

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR:-N/A}"
echo "Home directory: ${HOME}"
echo "Working directory before cd: $PWD"
echo "Current node: ${SLURM_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

eval "$(conda shell.bash hook)"
conda activate trafficllm

MM4FLOW_DIR="MM4flow"

SOURCE_DATASET_NAME="ITC_Net_A"
TARGET_DATASET_NAME="ITC_Net_D"

SOURCE_DATASET="finetune_dataset/${SOURCE_DATASET_NAME}"
TARGET_CSV="finetune_dataset/${TARGET_DATASET_NAME}/dataset.csv.gz"

VIEWS="ps,byte,mm"

PS_MODEL_TS="ps_60000/ps_${SOURCE_DATASET_NAME}_id"
BYTE_MODEL_TS="byte_60000/byte_${SOURCE_DATASET_NAME}_id"
MM_MODEL_TS="mm_60000/mm_${SOURCE_DATASET_NAME}_id"

OUTPUT_SUFFIX="_D_global_pseudo_residual_cls"
OUTPUT_DIR="outputs/mm4flow/global_pseudo_residual_${SOURCE_DATASET_NAME}_to_${TARGET_DATASET_NAME}_${VIEWS}"

# 可选：复用/保存 features cache
REUSE_FEATURE_CACHE_DIR=""
SAVE_FEATURE_CACHE_DIR=""

STEPS=2000
LR=1e-4
FEATURE_BATCH_SIZE=256
INFER_BATCH_SIZE=16
EVAL_EVERY=200
SELECT_BEST="weighted_f1"

ANCHOR_WEIGHT=1e-3
SHIFT_WEIGHT=1.0
SHIFT_RHO=0.5

ALPHA_MAX=0.5
RESIDUAL_RHO=1.0
MIN_COUNT=3

# 预算实验：1.0 表示使用全部 target；可以改成 0.5 / 0.25 / 0.1 做稳定性实验。
TARGET_BUDGET_RATIO=1.0
TARGET_BUDGET_SAMPLING="stratified"
TARGET_BUDGET_SEED=128

cd "${MM4FLOW_DIR}"

mkdir -p "finetune_log/${SOURCE_DATASET_NAME}"
mkdir -p model-classifier
mkdir -p "$(dirname "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Python: $(which python)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "========== Start Global-Pseudo Residual Classifier Fine-tune =========="
echo "SOURCE_DATASET=${SOURCE_DATASET}"
echo "TARGET_CSV=${TARGET_CSV}"
echo "VIEWS=${VIEWS}"
echo "PS_MODEL_TS=${PS_MODEL_TS}"
echo "BYTE_MODEL_TS=${BYTE_MODEL_TS}"
echo "MM_MODEL_TS=${MM_MODEL_TS}"
echo "OUTPUT_SUFFIX=${OUTPUT_SUFFIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ALPHA_MAX=${ALPHA_MAX}"
echo "RESIDUAL_RHO=${RESIDUAL_RHO}"
echo "MIN_COUNT=${MIN_COUNT}"
echo "TARGET_BUDGET_RATIO=${TARGET_BUDGET_RATIO}"
echo "TARGET_BUDGET_SAMPLING=${TARGET_BUDGET_SAMPLING}"
echo "TARGET_BUDGET_SEED=${TARGET_BUDGET_SEED}"
echo "REUSE_FEATURE_CACHE_DIR=${REUSE_FEATURE_CACHE_DIR:-unset}"
echo "SAVE_FEATURE_CACHE_DIR=${SAVE_FEATURE_CACHE_DIR:-unset}"

EXTRA_ARGS=""
if [ -n "${REUSE_FEATURE_CACHE_DIR}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --reuse_feature_cache_dir ${REUSE_FEATURE_CACHE_DIR}"
fi
if [ -n "${SAVE_FEATURE_CACHE_DIR}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --save_feature_cache_dir ${SAVE_FEATURE_CACHE_DIR}"
fi

python proj_classifier_global_pseudo_residual_finetune.py \
  --source_dataset "${SOURCE_DATASET}" \
  --target_csv "${TARGET_CSV}" \
  --views "${VIEWS}" \
  --ps_model_ts "${PS_MODEL_TS}" \
  --byte_model_ts "${BYTE_MODEL_TS}" \
  --mm_model_ts "${MM_MODEL_TS}" \
  --output_suffix "${OUTPUT_SUFFIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --adapt_strategy global_pseudo_residual \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --batch_size "${FEATURE_BATCH_SIZE}" \
  --infer_batch_size "${INFER_BATCH_SIZE}" \
  --eval_every "${EVAL_EVERY}" \
  --select_best "${SELECT_BEST}" \
  --anchor_weight "${ANCHOR_WEIGHT}" \
  --pseudo_residual_alpha_max "${ALPHA_MAX}" \
  --pseudo_residual_rho "${RESIDUAL_RHO}" \
  --pseudo_residual_min_count "${MIN_COUNT}" \
  --pseudo_residual_shift_rho "${SHIFT_RHO}" \
  --pseudo_residual_shift_weight "${SHIFT_WEIGHT}" \
  --target_budget_ratio "${TARGET_BUDGET_RATIO}" \
  --target_budget_sampling "${TARGET_BUDGET_SAMPLING}" \
  --target_budget_seed "${TARGET_BUDGET_SEED}" \
  --eval_target_after_train \
  ${EXTRA_ARGS}

echo "========== Finished Global-Pseudo Residual Fine-tune =========="
echo "Global-pseudo residual model_ts will be:"
echo "  ${PS_MODEL_TS}${OUTPUT_SUFFIX}"
echo "  ${BYTE_MODEL_TS}${OUTPUT_SUFFIX}"
echo "  ${MM_MODEL_TS}${OUTPUT_SUFFIX}"
echo "Diagnostics under: ${OUTPUT_DIR}"
echo "Summary:"
cat "${OUTPUT_DIR}/global_pseudo_residual_summary.json" || true
