#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

REPEATS="${REPEATS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
NUM_WARMUPS="${NUM_WARMUPS:-4}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32}"
MODEL="${MODEL:-facebook/opt-125m}"
TEMPERATURE="${TEMPERATURE:-0}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
RUN_ROOT="${RUN_ROOT:-/scratch/deepseek-prof/profiles/serve_repeated}"
LOG_ROOT="${LOG_ROOT:-/scratch/deepseek-prof/logs/serve_repeated}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

started_at="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="$RUN_ROOT/repeated_${started_at}_manifest.txt"

{
  echo "started_at=$started_at"
  echo "repeats=$REPEATS"
  echo "num_prompts=$NUM_PROMPTS"
  echo "num_warmups=$NUM_WARMUPS"
  echo "concurrencies=$CONCURRENCIES"
  echo "model=$MODEL"
  echo "temperature=$TEMPERATURE"
  echo "request_rate=$REQUEST_RATE"
} | tee "$manifest"

for repeat in $(seq 1 "$REPEATS"); do
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out_dir="$RUN_ROOT/${stamp}_repeat_${repeat}"
  log_dir="$LOG_ROOT/${stamp}_repeat_${repeat}"
  mkdir -p "$out_dir" "$log_dir"

  echo "repeat=$repeat out_dir=$out_dir log_dir=$log_dir" | tee -a "$manifest"

  MODEL="$MODEL" \
  NUM_PROMPTS="$NUM_PROMPTS" \
  NUM_WARMUPS="$NUM_WARMUPS" \
  CONCURRENCIES="$CONCURRENCIES" \
  TEMPERATURE="$TEMPERATURE" \
  REQUEST_RATE="$REQUEST_RATE" \
  OUT_DIR="$out_dir" \
  LOG_DIR="$log_dir" \
    /scratch/deepseek-prof/scripts/run_serve_realistic_matrix.sh \
    2>&1 | tee "$log_dir/matrix.log"
done

python /scratch/deepseek-prof/scripts/summarize_serve_results.py "$RUN_ROOT"/*/*.json \
  | tee "$RUN_ROOT/repeated_${started_at}_summary.md"

echo "finished_at=$(date -u +%Y%m%dT%H%M%SZ)" | tee -a "$manifest"
