#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

MODEL="${MODEL:-facebook/opt-125m}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-bench-model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18001}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
TEMPERATURE="${TEMPERATURE:-0}"

OUT_DIR="/scratch/deepseek-prof/profiles/serve_realistic"
NSYS_DIR="/scratch/deepseek-prof/profiles/nsys"
LOG_DIR="/scratch/deepseek-prof/logs/serve_realistic"
mkdir -p "$OUT_DIR" "$NSYS_DIR" "$LOG_DIR"

server_log="$LOG_DIR/nsys_decode_server_${MODEL//\//_}_${PORT}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

/usr/local/bin/nsys profile \
  --output "$NSYS_DIR/serve_decode_heavy_c${CONCURRENCY}_%h_%p" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --cuda-trace-scope=process-tree \
  --cuda-memory-usage=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$server_log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -200 "$server_log" >&2 || true
    exit 1
  fi
  sleep 1
done

vllm bench serve \
  --backend openai \
  --base-url "http://$HOST:$PORT" \
  --endpoint /v1/completions \
  --model "$SERVED_MODEL_NAME" \
  --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --random-range-ratio 0.20 \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups 4 \
  --request-rate inf \
  --max-concurrency "$CONCURRENCY" \
  --temperature "$TEMPERATURE" \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --save-detailed \
  --result-dir "$OUT_DIR" \
  --result-filename "nsys_decode_heavy_c${CONCURRENCY}_i128_o512_n${NUM_PROMPTS}.json"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
unset SERVER_PID

latest="$(ls -t "$NSYS_DIR"/serve_decode_heavy_c${CONCURRENCY}_*.nsys-rep | head -1)"
/usr/local/bin/nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$latest" \
  > "${latest%.nsys-rep}.stats.txt"
echo "$latest"
echo "${latest%.nsys-rep}.stats.txt"
