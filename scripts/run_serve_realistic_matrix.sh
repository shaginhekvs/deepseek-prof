#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

MODEL="${MODEL:-facebook/opt-125m}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-bench-model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
NUM_WARMUPS="${NUM_WARMUPS:-4}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32}"
TEMPERATURE="${TEMPERATURE:-0}"
PLOT_TIMELINE="${PLOT_TIMELINE:-0}"

OUT_DIR="/scratch/deepseek-prof/profiles/serve_realistic"
LOG_DIR="/scratch/deepseek-prof/logs/serve_realistic"
mkdir -p "$OUT_DIR" "$LOG_DIR"

server_log="$LOG_DIR/server_${MODEL//\//_}_${PORT}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$server_log" 2>&1 &
SERVER_PID=$!

echo "server_pid=$SERVER_PID"
echo "server_log=$server_log"

for _ in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited early; tailing log:" >&2
    tail -200 "$server_log" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "vLLM server did not become ready; tailing log:" >&2
  tail -200 "$server_log" >&2 || true
  exit 1
fi

run_case() {
  local name="$1"
  local input_len="$2"
  local output_len="$3"
  local range_ratio="$4"
  local concurrency="$5"
  local result_file="${name}_c${concurrency}_i${input_len}_o${output_len}_n${NUM_PROMPTS}.json"

  echo "running case=$name concurrency=$concurrency input=$input_len output=$output_len"
  vllm bench serve \
    --backend openai \
    --base-url "http://$HOST:$PORT" \
    --endpoint /v1/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio "$range_ratio" \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$concurrency" \
    --temperature "$TEMPERATURE" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "$OUT_DIR" \
    --result-filename "$result_file"

  if [[ "$PLOT_TIMELINE" == "1" ]]; then
    echo "timeline plotting is disabled in the default command because vllm[bench] extras are not installed"
  fi
}

for concurrency in $CONCURRENCIES; do
  run_case "prefill_heavy" 1536 32 0.15 "$concurrency"
  run_case "decode_heavy" 128 512 0.20 "$concurrency"
  run_case "mixed_chat" 768 256 0.35 "$concurrency"
done

python /scratch/deepseek-prof/scripts/summarize_serve_results.py "$OUT_DIR"/*.json \
  | tee "$OUT_DIR/summary.md"
